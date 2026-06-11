from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from dataloader import TEDataloader
from seg_models import SVGF16
from te_pretrain import DEFAULT_PRETRAIN_PATH
from utils import SegPyError, checkpoint_default_path, ensure_dir, require_torch, setup_logging, write_json_result


PROJECT_ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SVGF16 segmentation model from TeAiFlow .gt3 annotations.")
    parser.add_argument("--gt-dir", required=True, help="Directory containing TeAiFlow .gt3 annotations.")
    parser.add_argument("--image-dir", default=None, help="Source image directory. Defaults to gt_dir/../1/SrcImage.")
    parser.add_argument("--receptive-field", type=int, default=64, choices=[32, 64, 128, 256])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples-per-image", type=int, default=64)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--output", default=None, help="Checkpoint output path.")
    parser.add_argument("--pretrained-path", default=str(DEFAULT_PRETRAIN_PATH))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or cuda:0.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser


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

    dataset = TEDataloader(
        gt_dir=args.gt_dir,
        image_dir=args.image_dir,
        receptive_field=args.receptive_field,
        patch_size=args.patch_size,
        samples_per_image=args.samples_per_image,
        positive_ratio=args.positive_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = SVGF16(
        receptive_field=args.receptive_field,
        num_classes=args.num_classes,
        pretrained_path=args.pretrained_path,
        load_pretrained=not args.no_pretrained,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_path = Path(args.output) if args.output else checkpoint_default_path(PROJECT_ROOT, args.receptive_field)
    ensure_dir(output_path.parent)
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
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
            total_loss += float(loss.detach().cpu())
            steps += 1

        if steps <= 0:
            raise SegPyError("Training DataLoader produced no batches.")
        avg_loss = total_loss / steps
        logger.info("epoch=%s/%s loss=%.6f", epoch, args.epochs, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "class_name": "SVGF16",
                    "receptive_field": args.receptive_field,
                    "in_channels": 3,
                    "num_classes": args.num_classes,
                },
                "classes": ["BG", "defect"],
                "preprocessing": {
                    "input_scale": "0..1",
                    "patch_size": dataset.patch_size,
                    "positive_ratio": args.positive_ratio,
                },
                "training": {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "gt_dir": str(Path(args.gt_dir)),
                    "image_dir": str(dataset.image_dir),
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
        "samples": len(dataset.samples),
        "patchSize": dataset.patch_size,
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

