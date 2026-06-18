from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataloader import YOLOSegDataloader
from seg_models import build_svgf16_from_checkpoint
from utils import (
    SegPyError,
    ensure_dir,
    image_to_float_array,
    load_torch_checkpoint,
    receptive_field_to_stride,
    require_torch,
    setup_logging,
    stable_color,
    write_json_result,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SVGF16 on a YOLO segmentation dataset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-yaml", default=None)
    parser.add_argument("--yolo-image-dir", default=None)
    parser.add_argument("--yolo-label-dir", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", "--tile_size", dest="tile_size", type=int, default=512)
    parser.add_argument("--tile-overlap", "--tile_overlap", dest="tile_overlap", type=int, default=64)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--mask-thresh", "--mask_thresh", dest="mask_thresh", type=float, default=0.0)
    parser.add_argument("--max-images", "--max_images", dest="max_images", type=int, default=0)
    parser.add_argument("--start-index", "--start_index", dest="start_index", type=int, default=0)
    parser.add_argument("--max-vis", "--max_vis", dest="max_vis", type=int, default=24)
    parser.add_argument("--vis-tile-size", "--vis_tile_size", dest="vis_tile_size", type=int, default=256)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.dataset_yaml is not None and (args.yolo_image_dir is not None or args.yolo_label_dir is not None):
        raise SegPyError("Use either --dataset-yaml or --yolo-image-dir/--yolo-label-dir, not both.")
    if args.dataset_yaml is None and (args.yolo_image_dir is None or args.yolo_label_dir is None):
        raise SegPyError("YOLO evaluation requires --dataset-yaml or both --yolo-image-dir and --yolo-label-dir.")
    if args.tile_size <= 0:
        raise SegPyError(f"tile-size must be > 0, got {args.tile_size}")
    if args.tile_overlap < 0:
        raise SegPyError(f"tile-overlap must be >= 0, got {args.tile_overlap}")
    if args.tile_overlap >= args.tile_size:
        raise SegPyError(
            f"tile-overlap must be smaller than tile-size: tileOverlap={args.tile_overlap}, tileSize={args.tile_size}"
        )
    if args.batch_size <= 0:
        raise SegPyError(f"batch-size must be > 0, got {args.batch_size}")
    if not 0.0 <= args.mask_thresh <= 1.0:
        raise SegPyError(f"mask-thresh must be in [0,1], got {args.mask_thresh}")
    if args.max_images < 0:
        raise SegPyError(f"max-images must be >= 0, got {args.max_images}")
    if args.start_index < 0:
        raise SegPyError(f"start-index must be >= 0, got {args.start_index}")
    if args.max_vis < 0:
        raise SegPyError(f"max-vis must be >= 0, got {args.max_vis}")
    if args.vis_tile_size <= 0:
        raise SegPyError(f"vis-tile-size must be > 0, got {args.vis_tile_size}")


def _classes_from_checkpoint(checkpoint: dict[str, Any], num_classes: int) -> list[str]:
    raw_classes = checkpoint.get("classes")
    classes = [str(item) for item in raw_classes] if isinstance(raw_classes, list) else []
    while len(classes) < num_classes:
        classes.append(f"class_{len(classes)}")
    return classes[:num_classes]


def _select_samples(dataset: YOLOSegDataloader, start_index: int, max_images: int):
    samples = dataset.samples[start_index:]
    if max_images > 0:
        samples = samples[:max_images]
    if not samples:
        raise SegPyError(
            f"No samples selected for evaluation: startIndex={start_index}, maxImages={max_images}, "
            f"datasetSamples={len(dataset.samples)}"
        )
    return samples


def _tile_starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= 0:
        raise SegPyError(f"Invalid image dimension: {length}")
    if tile_size <= 0 or step <= 0:
        raise SegPyError(f"Invalid tile settings: tileSize={tile_size}, step={step}")
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size, 0) + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _valid_region(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
    half_overlap: int,
) -> tuple[int, int, int, int]:
    vx0 = x0 if x0 == 0 else x0 + half_overlap
    vy0 = y0 if y0 == 0 else y0 + half_overlap
    vx1 = x1 if x1 == width else x1 - half_overlap
    vy1 = y1 if y1 == height else y1 - half_overlap
    if vx0 >= vx1 or vy0 >= vy1:
        raise SegPyError(
            f"Invalid valid tile region: tile=({x0},{y0},{x1},{y1}), "
            f"valid=({vx0},{vy0},{vx1},{vy1}), image={width}x{height}"
        )
    return vx0, vy0, vx1, vy1


def _pad_tile(tile: np.ndarray, tile_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = tile.shape[:2]
    pad_h = tile_size - height
    pad_w = tile_size - width
    if pad_h < 0 or pad_w < 0:
        raise SegPyError(f"Tile is larger than tile_size: tile={width}x{height}, tileSize={tile_size}")
    if pad_h == 0 and pad_w == 0:
        return tile, (height, width)
    padded = np.pad(tile, [(0, pad_h), (0, pad_w), (0, 0)], mode="edge")
    return padded, (height, width)


def _predict_tile_batch(
    torch: Any,
    model: Any,
    tiles: list[np.ndarray],
    original_sizes: list[tuple[int, int]],
    device: str,
    mask_thresh: float,
) -> list[np.ndarray]:
    if not tiles:
        return []
    batch = np.stack([tile.transpose(2, 0, 1) for tile in tiles], axis=0)
    tensor = torch.from_numpy(batch).float().to(device)
    with torch.no_grad():
        logits = model(tensor).detach()
        if mask_thresh > 0.0:
            probs = torch.softmax(logits, dim=1)
            score, label = torch.max(probs, dim=1)
            label = torch.where((label > 0) & (score < mask_thresh), torch.zeros_like(label), label)
        else:
            label = torch.argmax(logits, dim=1)
        labels = label.detach().cpu().numpy().astype(np.uint8)
    del tensor, logits, label
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    results: list[np.ndarray] = []
    for index, (height, width) in enumerate(original_sizes):
        results.append(labels[index, :height, :width])
    return results


def predict_label_tiled(
    torch: Any,
    model: Any,
    image_array: np.ndarray,
    device: str,
    tile_size: int,
    tile_overlap: int,
    batch_size: int,
    mask_thresh: float,
    logger: logging.Logger,
) -> np.ndarray:
    height, width = image_array.shape[:2]
    step = tile_size - tile_overlap
    half_overlap = tile_overlap // 2
    y_starts = _tile_starts(height, tile_size, step)
    x_starts = _tile_starts(width, tile_size, step)
    pred_label = np.zeros((height, width), dtype=np.uint8)
    covered = np.zeros((height, width), dtype=np.bool_)
    tile_count = len(y_starts) * len(x_starts)
    logger.info("Tiled eval inference: image=%sx%s tile=%s overlap=%s batch=%s tiles=%s", width, height, tile_size, tile_overlap, batch_size, tile_count)

    batch_tiles: list[np.ndarray] = []
    batch_sizes: list[tuple[int, int]] = []
    batch_meta: list[tuple[int, int, int, int, int, int, int, int]] = []
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            vx0, vy0, vx1, vy1 = _valid_region(x0, y0, x1, y1, width, height, half_overlap)
            tile = image_array[y0:y1, x0:x1, :]
            padded, original_size = _pad_tile(tile, tile_size)
            batch_tiles.append(padded)
            batch_sizes.append(original_size)
            batch_meta.append((x0, y0, x1, y1, vx0, vy0, vx1, vy1))
            if len(batch_tiles) >= batch_size:
                _flush_tile_batch(torch, model, batch_tiles, batch_sizes, batch_meta, pred_label, covered, device, mask_thresh)
                batch_tiles, batch_sizes, batch_meta = [], [], []
    if batch_tiles:
        _flush_tile_batch(torch, model, batch_tiles, batch_sizes, batch_meta, pred_label, covered, device, mask_thresh)

    if not bool(covered.all()):
        missing = int((~covered).sum())
        raise SegPyError(f"Tiled prediction failed; uncovered pixels={missing}")
    return pred_label


def _flush_tile_batch(
    torch: Any,
    model: Any,
    batch_tiles: list[np.ndarray],
    batch_sizes: list[tuple[int, int]],
    batch_meta: list[tuple[int, int, int, int, int, int, int, int]],
    pred_label: np.ndarray,
    covered: np.ndarray,
    device: str,
    mask_thresh: float,
) -> None:
    labels = _predict_tile_batch(torch, model, batch_tiles, batch_sizes, device, mask_thresh)
    for label, meta in zip(labels, batch_meta):
        x0, y0, _x1, _y1, vx0, vy0, vx1, vy1 = meta
        sy0 = vy0 - y0
        sx0 = vx0 - x0
        sy1 = vy1 - y0
        sx1 = vx1 - x0
        pred_label[vy0:vy1, vx0:vx1] = label[sy0:sy1, sx0:sx1]
        covered[vy0:vy1, vx0:vx1] = True


def confusion_matrix(target: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    if target.shape != pred.shape:
        raise SegPyError(f"target/pred shape mismatch: target={target.shape}, pred={pred.shape}")
    target_max = int(target.max()) if target.size else 0
    pred_max = int(pred.max()) if pred.size else 0
    if target_max >= num_classes:
        raise SegPyError(f"Target label exceeds checkpoint num_classes: maxTarget={target_max}, numClasses={num_classes}")
    if pred_max >= num_classes:
        raise SegPyError(f"Prediction label exceeds checkpoint num_classes: maxPred={pred_max}, numClasses={num_classes}")
    flat = target.astype(np.int64).ravel() * num_classes + pred.astype(np.int64).ravel()
    counts = np.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape((num_classes, num_classes)).astype(np.int64)


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def metrics_from_confusion(cm: np.ndarray, classes: list[str]) -> dict[str, Any]:
    num_classes = cm.shape[0]
    total = int(cm.sum())
    correct = int(np.trace(cm))
    tp_fg = int(cm[1:, 1:].sum())
    fp_fg = int(cm[0, 1:].sum())
    fn_fg = int(cm[1:, 0].sum())

    per_class = []
    ious: list[float] = []
    for class_id in range(num_classes):
        tp = int(cm[class_id, class_id])
        target_count = int(cm[class_id, :].sum())
        pred_count = int(cm[:, class_id].sum())
        union = target_count + pred_count - tp
        iou = _safe_div(tp, union)
        dice = _safe_div(2 * tp, target_count + pred_count)
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
                "dice": _round_or_none(dice),
            }
        )

    return {
        "pixels": total,
        "pixel_acc": _round_or_none(_safe_div(correct, total)),
        "fg_iou": _round_or_none(_safe_div(tp_fg, tp_fg + fp_fg + fn_fg)),
        "fg_precision": _round_or_none(_safe_div(tp_fg, tp_fg + fp_fg)),
        "fg_recall": _round_or_none(_safe_div(tp_fg, tp_fg + fn_fg)),
        "fg_dice": _round_or_none(_safe_div(2 * tp_fg, 2 * tp_fg + fp_fg + fn_fg)),
        "mIoU": _round_or_none(float(np.mean(ious)) if ious else None),
        "target_pos_pixels": int(cm[1:, :].sum()),
        "pred_pos_pixels": int(cm[:, 1:].sum()),
        "tp_fg_pixels": tp_fg,
        "fp_pixels": fp_fg,
        "fn_pixels": fn_fg,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _label_to_rgb(label: np.ndarray, num_classes: int) -> np.ndarray:
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id in range(1, num_classes):
        rgb[label == class_id] = stable_color(class_id)
    return rgb


def _overlay_prediction(image: Image.Image, pred: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    mask = pred > 0
    red = np.asarray((255, 64, 64), dtype=np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + red * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def _choose_crop_box(target: np.ndarray, pred: np.ndarray, crop_size: int) -> tuple[int, int, int, int]:
    height, width = target.shape
    crop_w = min(crop_size, width)
    crop_h = min(crop_size, height)
    error = ((target > 0) != (pred > 0)) | ((target > 0) & (pred > 0) & (target != pred))
    points = np.argwhere(error)
    if points.size == 0:
        points = np.argwhere((target > 0) | (pred > 0))
    if points.size > 0:
        y, x = points[len(points) // 2]
    else:
        y, x = height // 2, width // 2
    x0 = int(np.clip(int(x) - crop_w // 2, 0, max(0, width - crop_w)))
    y0 = int(np.clip(int(y) - crop_h // 2, 0, max(0, height - crop_h)))
    return x0, y0, x0 + crop_w, y0 + crop_h


def _panel_with_title(image: Image.Image, title: str, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height + 18), (0, 0, 0))
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    panel.paste(resized, (0, 18))
    draw = ImageDraw.Draw(panel)
    draw.text((3, 2), title, fill=(255, 255, 255), font=ImageFont.load_default())
    return panel


def _make_sample_block(
    image: Image.Image,
    target: np.ndarray,
    pred: np.ndarray,
    sample_name: str,
    num_classes: int,
    crop_size: int,
) -> Image.Image:
    x0, y0, x1, y1 = _choose_crop_box(target, pred, crop_size)
    image_crop = image.crop((x0, y0, x1, y1))
    target_crop = target[y0:y1, x0:x1]
    pred_crop = pred[y0:y1, x0:x1]
    gt_image = Image.fromarray(_label_to_rgb(target_crop, num_classes), mode="RGB")
    pred_image = Image.fromarray(_label_to_rgb(pred_crop, num_classes), mode="RGB")
    overlay = _overlay_prediction(image_crop, pred_crop)

    panel_w = crop_size
    panel_h = crop_size
    block = Image.new("RGB", (panel_w * 2, (panel_h + 18) * 2 + 16), (0, 0, 0))
    panels = [
        _panel_with_title(image_crop, "Image", panel_w, panel_h),
        _panel_with_title(gt_image, "GT", panel_w, panel_h),
        _panel_with_title(pred_image, "Pred", panel_w, panel_h),
        _panel_with_title(overlay, "Overlay", panel_w, panel_h),
    ]
    block.paste(panels[0], (0, 0))
    block.paste(panels[1], (panel_w, 0))
    block.paste(panels[2], (0, panel_h + 18))
    block.paste(panels[3], (panel_w, panel_h + 18))
    draw = ImageDraw.Draw(block)
    label = sample_name[:48]
    draw.text((3, block.height - 14), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return block


def _save_mosaics(blocks: list[Image.Image], output_dir: Path, max_vis: int) -> list[str]:
    if max_vis <= 0 or not blocks:
        return []
    output_paths: list[str] = []
    blocks_per_mosaic = min(max_vis, 24)
    for mosaic_index, start in enumerate(range(0, len(blocks), blocks_per_mosaic)):
        chunk = blocks[start : start + blocks_per_mosaic]
        cols = min(4, len(chunk))
        rows = math.ceil(len(chunk) / cols)
        block_w, block_h = chunk[0].size
        mosaic = Image.new("RGB", (cols * block_w, rows * block_h), (0, 0, 0))
        for item_index, block in enumerate(chunk):
            col = item_index % cols
            row = item_index // cols
            mosaic.paste(block, (col * block_w, row * block_h))
        path = output_dir / f"mosaic_{mosaic_index:03d}.jpg"
        mosaic.save(path, quality=92)
        output_paths.append(str(path))
    return output_paths


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    torch = require_torch()
    logger = logging.getLogger("segpy.eval")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        device = "cpu"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise SegPyError(f"Checkpoint does not exist or is not a file: {checkpoint_path}")
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    model = build_svgf16_from_checkpoint(checkpoint, device=device)
    model.eval()
    config = checkpoint.get("model_config", {})
    if not isinstance(config, dict):
        raise SegPyError("Checkpoint model_config must be a dict.")
    num_classes = int(config.get("num_classes", 2))
    receptive_field = int(config.get("receptive_field", 64))
    classes = _classes_from_checkpoint(checkpoint, num_classes)
    stride = receptive_field_to_stride(receptive_field)
    if args.tile_size % stride != 0:
        raise SegPyError(f"tile-size must be divisible by model stride={stride}, got tileSize={args.tile_size}")

    dataset = YOLOSegDataloader(
        dataset_yaml=args.dataset_yaml,
        image_dir=args.yolo_image_dir,
        label_dir=args.yolo_label_dir,
        split=args.split,
        receptive_field=receptive_field,
        samples_per_image=1,
        validate_masks=False,
        cache_samples=False,
    )
    if dataset.max_label >= num_classes:
        raise SegPyError(
            f"Dataset labels exceed checkpoint output classes: maxLabel={dataset.max_label}, numClasses={num_classes}"
        )
    selected_samples = _select_samples(dataset, args.start_index, args.max_images)

    output_dir = ensure_dir(args.output_dir)
    global_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_image_rows: list[dict[str, Any]] = []
    vis_blocks: list[Image.Image] = []

    for index, sample in enumerate(selected_samples, start=1):
        logger.info("Evaluating %s/%s: %s", index, len(selected_samples), sample.name)
        image, target = dataset.load_full_sample(sample)
        image_array = image_to_float_array(image)
        pred = predict_label_tiled(
            torch=torch,
            model=model,
            image_array=image_array,
            device=device,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            batch_size=args.batch_size,
            mask_thresh=args.mask_thresh,
            logger=logger,
        )
        cm = confusion_matrix(target, pred, num_classes)
        global_cm += cm
        metrics = metrics_from_confusion(cm, classes)
        row = {
            "image": sample.name,
            "image_path": str(sample.image_path),
            "label_path": str(sample.label_path),
            "width": int(image.width),
            "height": int(image.height),
        }
        row.update({key: value for key, value in metrics.items() if key not in {"per_class", "confusion_matrix"}})
        per_image_rows.append(row)

        if len(vis_blocks) < args.max_vis:
            vis_blocks.append(_make_sample_block(image, target, pred, sample.name, num_classes, args.vis_tile_size))

        dataset.clear_cache()
        del image, target, image_array, pred, cm, metrics
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary_metrics = metrics_from_confusion(global_cm, classes)
    per_class_rows = summary_metrics["per_class"]
    per_image_csv = output_dir / "per_image_metrics.csv"
    per_class_csv = output_dir / "per_class_metrics.csv"
    _write_csv(
        per_image_csv,
        per_image_rows,
        [
            "image",
            "image_path",
            "label_path",
            "width",
            "height",
            "pixels",
            "pixel_acc",
            "fg_iou",
            "fg_precision",
            "fg_recall",
            "fg_dice",
            "mIoU",
            "target_pos_pixels",
            "pred_pos_pixels",
            "tp_fg_pixels",
            "fp_pixels",
            "fn_pixels",
        ],
    )
    _write_csv(
        per_class_csv,
        per_class_rows,
        ["class_id", "class_name", "target_pixels", "pred_pixels", "tp_pixels", "precision", "recall", "iou", "dice"],
    )
    mosaic_paths = _save_mosaics(vis_blocks, output_dir, args.max_vis)
    payload = {
        "ok": True,
        "checkpoint": str(checkpoint_path),
        "dataset": {
            "type": "yolo",
            "split": args.split,
            "image_dir": str(dataset.image_dir),
            "label_dir": str(dataset.label_dir),
            "selected_count": len(selected_samples),
            "total_count": len(dataset.samples),
            "start_index": args.start_index,
            "max_images": args.max_images,
        },
        "model": {
            "receptive_field": receptive_field,
            "stride": stride,
            "num_classes": num_classes,
            "classes": classes,
        },
        "inference": {
            "device": device,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "batch_size": args.batch_size,
            "mask_thresh": args.mask_thresh,
        },
        "summary": summary_metrics,
        "files": {
            "summary": str(output_dir / "summary.json"),
            "per_image_metrics": str(per_image_csv),
            "per_class_metrics": str(per_class_csv),
            "mosaics": mosaic_paths,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
        logging.getLogger("segpy.eval").exception("YOLO segmentation evaluation failed")
        write_json_result({"ok": False, "errorMessage": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
