from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from dataloader import TEDataloader, YOLOSegDataloader
from seg_models import SVGF16
from te_pretrain import DEFAULT_PRETRAIN_PATH
from utils import (
    SegPyError,
    checkpoint_default_path,
    ensure_dir,
    load_torch_checkpoint,
    require_torch,
    setup_logging,
    write_json_result,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SVGF16 segmentation model from .gt3 or YOLO segmentation data.")
    parser.add_argument("--dataset-type", default="gt3", choices=["gt3", "yolo"])
    parser.add_argument("--gt-dir", default=None, help="Directory containing TeAiFlow .gt3 annotations.")
    parser.add_argument("--image-dir", default=None, help="Source image directory. Defaults to gt_dir/../1/SrcImage.")
    parser.add_argument("--dataset-yaml", default=None, help="YOLO dataset.yaml path.")
    parser.add_argument("--yolo-image-dir", default=None, help="YOLO image directory.")
    parser.add_argument("--yolo-label-dir", default=None, help="YOLO label directory.")
    parser.add_argument("--split", default="train", help="YOLO dataset.yaml split name.")
    parser.add_argument("--receptive-field", type=int, default=64, choices=[32, 64, 128, 256])
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples-per-image", type=int, default=64)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--output", default=None, help="Checkpoint output path.")
    parser.add_argument("--pretrain", default=None, help="Path to a PyTorch .pt checkpoint or state_dict.")
    parser.add_argument("--pretrained-path", default=str(DEFAULT_PRETRAIN_PATH))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or cuda:0.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(isinstance(key, str) and key.startswith("module.") for key in keys):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_pt_pretrain_weights(model: Any, pretrain_path: str | Path, logger: logging.Logger) -> dict[str, Any]:
    path = Path(pretrain_path)
    if not path.exists():
        raise SegPyError(f"PyTorch pretrain checkpoint does not exist: {path}")
    if not path.is_file():
        raise SegPyError(f"PyTorch pretrain checkpoint path is not a file: {path}")

    checkpoint = load_torch_checkpoint(path, map_location="cpu")
    state_key = "raw_state_dict"
    if "model_state_dict" in checkpoint:
        raw_state = checkpoint["model_state_dict"]
        state_key = "model_state_dict"
    elif "state_dict" in checkpoint:
        raw_state = checkpoint["state_dict"]
        state_key = "state_dict"
    else:
        raw_state = checkpoint
    if not isinstance(raw_state, dict):
        raise SegPyError(f"PyTorch pretrain checkpoint does not contain a state_dict: {path}")
    state_dict = _strip_module_prefix(raw_state)
    model_state = model.state_dict()

    compatible_state: dict[str, Any] = {}
    skipped_missing = 0
    skipped_shape = 0
    skipped_non_tensor = 0
    skipped_examples: list[str] = []

    for name, tensor in state_dict.items():
        if name not in model_state:
            skipped_missing += 1
            if len(skipped_examples) < 8:
                skipped_examples.append(f"{name}: not in current model")
            continue
        if not hasattr(tensor, "shape"):
            skipped_non_tensor += 1
            if len(skipped_examples) < 8:
                skipped_examples.append(f"{name}: not a tensor")
            continue
        if tuple(tensor.shape) != tuple(model_state[name].shape):
            skipped_shape += 1
            if len(skipped_examples) < 8:
                skipped_examples.append(
                    f"{name}: shape mismatch pretrain={tuple(tensor.shape)} model={tuple(model_state[name].shape)}"
                )
            continue
        compatible_state[name] = tensor

    if not compatible_state:
        raise SegPyError(
            f"No compatible tensors were loaded from PyTorch pretrain checkpoint: {path}. "
            f"skippedMissing={skipped_missing}, skippedShape={skipped_shape}, skippedNonTensor={skipped_non_tensor}"
        )

    incompatible = model.load_state_dict(compatible_state, strict=False)
    missing_after_load = len(incompatible.missing_keys)
    unexpected_after_load = len(incompatible.unexpected_keys)
    report = {
        "type": "pt",
        "path": str(path),
        "state_key": state_key,
        "loaded": len(compatible_state),
        "available": len(state_dict),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "skipped_non_tensor": skipped_non_tensor,
        "model_missing_after_load": missing_after_load,
        "unexpected_after_load": unexpected_after_load,
        "skipped_examples": skipped_examples,
    }
    logger.info(
        "PyTorch pretrain load result: loaded=%s available=%s skipped_missing=%s skipped_shape=%s "
        "skipped_non_tensor=%s path=%s",
        report["loaded"],
        report["available"],
        skipped_missing,
        skipped_shape,
        skipped_non_tensor,
        path,
    )
    if skipped_examples:
        logger.info("PyTorch pretrain skipped examples: %s", skipped_examples)
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    from torch.utils.data import DataLoader

    logger = logging.getLogger("segpy.train")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        device = "cpu"

    if args.epochs <= 0:
        raise SegPyError(f"epochs must be > 0, got {args.epochs}")
    if args.batch_size <= 0:
        raise SegPyError(f"batch-size must be > 0, got {args.batch_size}")
    if args.lr <= 0:
        raise SegPyError(f"lr must be > 0, got {args.lr}")
    if args.pretrain is not None and args.no_pretrained:
        raise SegPyError("Use either --pretrain or --no-pretrained, not both.")

    if args.dataset_type == "gt3":
        if args.gt_dir is None:
            raise SegPyError("--gt-dir is required when --dataset-type gt3")
        dataset = TEDataloader(
            gt_dir=args.gt_dir,
            image_dir=args.image_dir,
            receptive_field=args.receptive_field,
            patch_size=args.patch_size,
            samples_per_image=args.samples_per_image,
            positive_ratio=args.positive_ratio,
        )
    else:
        dataset = YOLOSegDataloader(
            dataset_yaml=args.dataset_yaml,
            image_dir=args.yolo_image_dir,
            label_dir=args.yolo_label_dir,
            split=args.split,
            receptive_field=args.receptive_field,
            patch_size=args.patch_size,
            samples_per_image=args.samples_per_image,
            positive_ratio=args.positive_ratio,
        )
    inferred_num_classes = int(dataset.num_classes)
    if args.num_classes is not None:
        if args.num_classes < inferred_num_classes:
            raise SegPyError(
                f"--num-classes is smaller than dataset labels require: "
                f"numClasses={args.num_classes}, required={inferred_num_classes}"
            )
        num_classes = int(args.num_classes)
        classes = list(dataset.classes)
        while len(classes) < num_classes:
            classes.append(f"class_{len(classes)}")
    else:
        num_classes = inferred_num_classes
        classes = list(dataset.classes)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = SVGF16(
        receptive_field=args.receptive_field,
        num_classes=num_classes,
        pretrained_path=args.pretrained_path,
        load_pretrained=(not args.no_pretrained and args.pretrain is None),
    ).to(device)
    if args.pretrain is not None:
        model.pretrain_report = load_pt_pretrain_weights(model, args.pretrain, logger)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_path = Path(args.output) if args.output else checkpoint_default_path(PROJECT_ROOT, args.receptive_field)
    ensure_dir(output_path.parent)
    best_loss = float("inf")
    best_epoch = -1
    last_pos_acc = float("nan")
    last_neg_acc = float("nan")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pos_correct = 0
        pos_total = 0
        neg_correct = 0
        neg_total = 0
        steps = 0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                raise SegPyError(
                    f"Model output size differs from mask size: logits={tuple(logits.shape)} masks={tuple(masks.shape)}"
                )
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits.detach(), dim=1)
                target_pos = masks > 0
                target_neg = masks == 0
                pred_pos = pred > 0
                pred_neg = pred == 0
                pos_correct += int((pred_pos & target_pos).sum().item())
                pos_total += int(target_pos.sum().item())
                neg_correct += int((pred_neg & target_neg).sum().item())
                neg_total += int(target_neg.sum().item())

            total_loss += float(loss.detach().cpu())
            steps += 1

        if steps <= 0:
            raise SegPyError("Training DataLoader produced no batches.")
        avg_loss = total_loss / steps
        if pos_total > 0:
            last_pos_acc = pos_correct / pos_total
        else:
            last_pos_acc = float("nan")
            logger.warning("No positive foreground pixels were seen in epoch=%s; pos_acc=nan", epoch)
        if neg_total > 0:
            last_neg_acc = neg_correct / neg_total
        else:
            last_neg_acc = float("nan")
            logger.warning("No background pixels were seen in epoch=%s; neg_acc=nan", epoch)
        logger.info(
            "epoch=%s/%s loss=%.6f pos_acc=%.4f neg_acc=%.4f",
            epoch,
            args.epochs,
            avg_loss,
            last_pos_acc,
            last_neg_acc,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "class_name": "SVGF16",
                    "receptive_field": args.receptive_field,
                    "in_channels": 3,
                    "num_classes": num_classes,
                },
                "classes": classes,
                "preprocessing": {
                    "input_scale": "0..1",
                    "patch_size": dataset.patch_size,
                    "positive_ratio": args.positive_ratio,
                },
                "training": {
                    "dataset_type": args.dataset_type,
                    "class_mode": "multiclass",
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "gt_dir": str(Path(args.gt_dir)) if args.gt_dir is not None else None,
                    "image_dir": str(dataset.image_dir),
                    "label_dir": str(getattr(dataset, "label_dir", "")),
                    "dataset_yaml": str(Path(args.dataset_yaml)) if args.dataset_yaml is not None else None,
                    "samples": len(dataset.samples),
                    "samples_per_image": args.samples_per_image,
                },
                "pretrain_report": getattr(model, "pretrain_report", {}),
            }
            torch.save(checkpoint, output_path)

    return {
        "ok": True,
        "checkpoint": str(output_path),
        "bestLoss": best_loss,
        "bestEpoch": best_epoch,
        "lastPosAcc": last_pos_acc,
        "lastNegAcc": last_neg_acc,
        "samples": len(dataset.samples),
        "patchSize": dataset.patch_size,
        "numClasses": num_classes,
        "classes": classes,
        "datasetType": args.dataset_type,
        "pretrainReport": getattr(model, "pretrain_report", {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        write_json_result(train(args))
        return 0
    except Exception as exc:
        logging.getLogger("segpy.train").exception("Training failed")
        write_json_result({"ok": False, "errorMessage": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
