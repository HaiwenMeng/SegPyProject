from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataloader import TEDataloader, YOLOSegDataloader
from instance_postprocess import postprocess_logits
from predict import _classes_from_checkpoint, _infer_logits, _resolve_tile_settings
from seg_models import build_svgf16_from_checkpoint
from utils import (
    SegPyError,
    ensure_dir,
    image_to_float_array,
    load_torch_checkpoint,
    receptive_field_to_stride,
    require_torch,
    setup_logging,
    str2bool,
    write_json_result,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained SVGF16 checkpoint on full annotated images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-type", default="gt3", choices=["gt3", "yolo"])
    parser.add_argument("--gt-dir", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--dataset-yaml", default=None)
    parser.add_argument("--yolo-image-dir", default=None)
    parser.add_argument("--yolo-label-dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(PROJECT_ROOT / "outputs" / "eval_full"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", "--max_det", dest="max_det", type=int, default=1000)
    parser.add_argument("--min-pixel", "--min_pixel", "--min-pxiel", dest="min_pixel", type=int, default=1)
    parser.add_argument("--mask-thresh", "--mask_thresh", dest="mask_thresh", type=float, default=0.0)
    parser.add_argument("--eval-post", "--eval_post", dest="eval_post", type=str2bool, default=False)
    parser.add_argument("--validate-masks", "--validate_masks", dest="validate_masks", type=str2bool, default=False)
    parser.add_argument("--save-images", "--save_images", dest="save_images", type=str2bool, default=False)
    parser.add_argument("--max-images", "--max_images", dest="max_images", type=int, default=0)
    parser.add_argument("--start-index", "--start_index", dest="start_index", type=int, default=0)
    parser.add_argument("--tile", action="store_true", default=True, help="Enable tiled inference for very large images.")
    parser.add_argument("--tile-size", "--tile_size", dest="tile_size", type=int, default=None)
    parser.add_argument("--tile-overlap", "--tile_overlap", dest="tile_overlap", type=int, default=None)
    parser.add_argument("--no-tile", "--no_tile", dest="no_tile", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def build_dataset(args: argparse.Namespace, receptive_field: int):
    validate_masks = bool(args.validate_masks or args.dataset_type == "gt3")
    if args.dataset_type == "gt3":
        if args.gt_dir is None:
            raise SegPyError("--gt-dir is required when --dataset-type gt3")
        return TEDataloader(
            gt_dir=args.gt_dir,
            image_dir=args.image_dir,
            receptive_field=receptive_field,
            samples_per_image=1,
            positive_ratio=0.5,
            validate_masks=validate_masks,
            cache_samples=False,
        )
    return YOLOSegDataloader(
        dataset_yaml=args.dataset_yaml,
        image_dir=args.yolo_image_dir,
        label_dir=args.yolo_label_dir,
        split=args.split,
        receptive_field=receptive_field,
        samples_per_image=1,
        positive_ratio=0.5,
        validate_masks=validate_masks,
        cache_samples=False,
    )


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def confusion_matrix(target: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    if target.shape != pred.shape:
        raise SegPyError(f"target/pred shape mismatch: target={target.shape}, pred={pred.shape}")
    if target.ndim != 2:
        raise SegPyError(f"target/pred must be 2D label maps, got target shape={target.shape}")
    if num_classes <= 1:
        raise SegPyError(f"num_classes must be > 1, got {num_classes}")
    target_max = int(target.max()) if target.size else 0
    pred_max = int(pred.max()) if pred.size else 0
    if target_max >= num_classes:
        raise SegPyError(f"Target label exceeds checkpoint num_classes: maxTarget={target_max}, numClasses={num_classes}")
    if pred_max >= num_classes:
        raise SegPyError(f"Prediction label exceeds checkpoint num_classes: maxPred={pred_max}, numClasses={num_classes}")

    flat = target.astype(np.int64).ravel() * num_classes + pred.astype(np.int64).ravel()
    counts = np.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape((num_classes, num_classes)).astype(np.int64)


def metrics_from_confusion(cm: np.ndarray, classes: list[str]) -> dict[str, Any]:
    num_classes = cm.shape[0]
    total = int(cm.sum())
    exact_correct = int(np.trace(cm))

    tp_fg = int(cm[1:, 1:].sum())
    fp_fg = int(cm[0, 1:].sum())
    fn_fg = int(cm[1:, 0].sum())
    tn_bg = int(cm[0, 0])
    target_pos = int(cm[1:, :].sum())
    target_neg = int(cm[0, :].sum())
    pred_pos = int(cm[:, 1:].sum())
    pred_neg = int(cm[:, 0].sum())

    per_class = []
    ious: list[float] = []
    for class_id in range(num_classes):
        tp = int(cm[class_id, class_id])
        target_count = int(cm[class_id, :].sum())
        pred_count = int(cm[:, class_id].sum())
        union = target_count + pred_count - tp
        iou = _safe_div(tp, union)
        if union > 0:
            ious.append(float(iou))
        per_class.append(
            {
                "class_id": class_id,
                "class_name": classes[class_id] if class_id < len(classes) else f"class_{class_id}",
                "target_pixels": target_count,
                "pred_pixels": pred_count,
                "tp_pixels": tp,
                "precision": _round_or_none(_safe_div(tp, pred_count)),
                "recall": _round_or_none(_safe_div(tp, target_count)),
                "iou": _round_or_none(iou),
            }
        )

    return {
        "pixels": total,
        "exact_acc": _round_or_none(_safe_div(exact_correct, total)),
        "pos_acc": _round_or_none(_safe_div(tp_fg, target_pos)),
        "neg_acc": _round_or_none(_safe_div(tn_bg, target_neg)),
        "fg_precision": _round_or_none(_safe_div(tp_fg, tp_fg + fp_fg)),
        "fg_recall": _round_or_none(_safe_div(tp_fg, tp_fg + fn_fg)),
        "fg_iou": _round_or_none(_safe_div(tp_fg, tp_fg + fp_fg + fn_fg)),
        "miou": _round_or_none(float(np.mean(ious)) if ious else None),
        "target_pos_pixels": target_pos,
        "target_neg_pixels": target_neg,
        "pred_pos_pixels": pred_pos,
        "pred_neg_pixels": pred_neg,
        "tp_fg_pixels": tp_fg,
        "tn_bg_pixels": tn_bg,
        "fp_pixels": fp_fg,
        "fn_pixels": fn_fg,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def raw_argmax_label(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 3:
        raise SegPyError(f"logits must have shape [C,H,W], got {logits.shape}")
    if not np.all(np.isfinite(logits)):
        raise SegPyError("logits contain NaN or Inf values.")
    return np.argmax(logits, axis=0).astype(np.uint8)


def save_error_map(path: Path, target: np.ndarray, pred: np.ndarray) -> None:
    if target.shape != pred.shape:
        raise SegPyError(f"Cannot save error map for mismatched shapes: target={target.shape}, pred={pred.shape}")
    target_pos = target > 0
    pred_pos = pred > 0
    image = np.zeros((*target.shape, 3), dtype=np.uint8)
    image[(~target_pos) & (~pred_pos)] = (16, 16, 16)
    image[target_pos & pred_pos] = (0, 180, 0)
    image[(~target_pos) & pred_pos] = (255, 0, 0)
    image[target_pos & (~pred_pos)] = (0, 80, 255)
    image[(target_pos & pred_pos) & (target != pred)] = (255, 220, 0)
    Image.fromarray(image, mode="RGB").save(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image",
        "mode",
        "pixels",
        "exact_acc",
        "pos_acc",
        "neg_acc",
        "fg_precision",
        "fg_recall",
        "fg_iou",
        "miou",
        "target_pos_pixels",
        "pred_pos_pixels",
        "fp_pixels",
        "fn_pixels",
        "instance_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    logger = logging.getLogger("segpy.eval_full")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        device = "cpu"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SegPyError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    model = build_svgf16_from_checkpoint(checkpoint, device=device)
    model.eval()
    config = checkpoint.get("model_config", {})
    if not isinstance(config, dict):
        raise SegPyError("Checkpoint model_config must be a dict.")
    receptive_field = int(config.get("receptive_field", 64))
    num_classes = int(config.get("num_classes", 2))
    classes = _classes_from_checkpoint(checkpoint, num_classes)
    stride = receptive_field_to_stride(receptive_field)
    tile_size, tile_overlap, no_tile = _resolve_tile_settings(checkpoint, receptive_field, stride, args)
    dataset = build_dataset(args, receptive_field)
    if dataset.max_label >= num_classes:
        raise SegPyError(
            f"Dataset labels exceed checkpoint output classes: maxLabel={dataset.max_label}, numClasses={num_classes}"
        )

    output_dir = ensure_dir(args.output_dir)
    image_dir = ensure_dir(output_dir / "images") if args.save_images else None

    raw_cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    post_cm_total = np.zeros((num_classes, num_classes), dtype=np.int64) if args.eval_post else None
    image_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    if args.start_index < 0:
        raise SegPyError(f"start-index must be >= 0, got {args.start_index}")
    if args.max_images < 0:
        raise SegPyError(f"max-images must be >= 0, got {args.max_images}")
    samples = dataset.samples[args.start_index :]
    if args.max_images > 0:
        samples = samples[: args.max_images]
    if not samples:
        raise SegPyError(
            f"No samples selected for evaluation: startIndex={args.start_index}, maxImages={args.max_images}, "
            f"datasetSamples={len(dataset.samples)}"
        )

    for index, sample in enumerate(samples, start=1):
        logger.info("Evaluating image %s/%s: %s", index, len(samples), sample.name)
        image, target_mask = dataset.load_full_sample(sample)
        array = image_to_float_array(image)
        logits, inference_mode = _infer_logits(
            torch=torch,
            model=model,
            array=array,
            num_classes=num_classes,
            stride=stride,
            device=device,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            no_tile=no_tile,
            logger=logger,
        )
        if logits.shape[-2:] != target_mask.shape:
            raise SegPyError(
                f"Logits/mask shape mismatch for {sample.name}: logits={logits.shape}, mask={target_mask.shape}"
            )

        raw_label = raw_argmax_label(logits)
        post_result = None
        post_label = None
        if args.eval_post:
            post_result = postprocess_logits(
                logits=logits,
                classes=classes,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                min_pixel=args.min_pixel,
                mask_thresh=args.mask_thresh,
            )
            post_label = post_result.label_map.astype(np.uint8)

        raw_cm = confusion_matrix(target_mask, raw_label, num_classes)
        raw_cm_total += raw_cm
        raw_metrics = metrics_from_confusion(raw_cm, classes)
        post_metrics = None
        if post_label is not None and post_cm_total is not None:
            post_cm = confusion_matrix(target_mask, post_label, num_classes)
            post_cm_total += post_cm
            post_metrics = metrics_from_confusion(post_cm, classes)

        files: dict[str, str] = {}
        if image_dir is not None:
            gt_path = image_dir / f"{sample.name}_gt_label.png"
            raw_path = image_dir / f"{sample.name}_raw_label.png"
            raw_error_path = image_dir / f"{sample.name}_raw_error.png"
            Image.fromarray(target_mask.astype(np.uint8), mode="L").save(gt_path)
            Image.fromarray(raw_label.astype(np.uint8), mode="L").save(raw_path)
            save_error_map(raw_error_path, target_mask, raw_label)
            files = {
                "gt_label": str(gt_path),
                "raw_label": str(raw_path),
                "raw_error": str(raw_error_path),
            }
            if post_label is not None:
                post_path = image_dir / f"{sample.name}_post_label.png"
                post_error_path = image_dir / f"{sample.name}_post_error.png"
                Image.fromarray(post_label.astype(np.uint8), mode="L").save(post_path)
                save_error_map(post_error_path, target_mask, post_label)
                files["post_label"] = str(post_path)
                files["post_error"] = str(post_error_path)

        image_result = {
            "name": sample.name,
            "image": str(sample.image_path),
            "label": str(sample.label_path),
            "width": int(image.width),
            "height": int(image.height),
            "inference_mode": inference_mode,
            "post_instance_count": len(post_result.instances) if post_result is not None else None,
            "raw": raw_metrics,
            "post": post_metrics,
            "files": files,
        }
        image_results.append(image_result)
        metric_rows = [("raw", raw_metrics)]
        if post_metrics is not None:
            metric_rows.append(("post", post_metrics))
        for mode, metrics in metric_rows:
            instance_count = len(post_result.instances) if mode == "post" and post_result is not None else None
            row = {"image": sample.name, "mode": mode, "instance_count": instance_count}
            row.update({key: value for key, value in metrics.items() if key != "per_class" and key != "confusion_matrix"})
            csv_rows.append(row)

        logger.info(
            "image=%s raw_fg_iou=%s raw_precision=%s raw_recall=%s post_fg_iou=%s post_precision=%s "
            "post_recall=%s instances=%s",
            sample.name,
            raw_metrics["fg_iou"],
            raw_metrics["fg_precision"],
            raw_metrics["fg_recall"],
            post_metrics["fg_iou"] if post_metrics is not None else None,
            post_metrics["fg_precision"] if post_metrics is not None else None,
            post_metrics["fg_recall"] if post_metrics is not None else None,
            len(post_result.instances) if post_result is not None else None,
        )
        dataset.clear_cache()
        del image, target_mask, array, logits, raw_label, raw_cm, raw_metrics, post_label, post_result, post_metrics
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary = {
        "raw": metrics_from_confusion(raw_cm_total, classes),
        "post": metrics_from_confusion(post_cm_total, classes) if post_cm_total is not None else None,
    }
    payload = {
        "ok": True,
        "checkpoint": str(checkpoint_path),
        "dataset_type": args.dataset_type,
        "num_classes": num_classes,
        "classes": classes,
        "sample_count": len(dataset.samples),
        "inference": {
            "stride": stride,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "tile_enabled": not no_tile,
            "eval_post": args.eval_post,
            "validate_masks": args.validate_masks,
            "save_images": args.save_images,
            "start_index": args.start_index,
            "max_images": args.max_images,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "min_pixel": args.min_pixel,
            "mask_thresh": args.mask_thresh,
        },
        "summary": summary,
        "images": image_results,
    }

    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "per_image_metrics.csv"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, csv_rows)
    payload["files"] = {"summary": str(summary_path), "csv": str(csv_path)}
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_arg_parser()
        args = parser.parse_args(argv)
        setup_logging(args.log_level)
        write_json_result(evaluate(args))
        return 0
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        write_json_result({"ok": False, "errorMessage": f"Invalid command line arguments. Exit code: {exc.code}"})
        return int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        logging.getLogger("segpy.eval_full").exception("Full-image evaluation failed")
        write_json_result({"ok": False, "errorMessage": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
