from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from seg_models import build_svgf16_from_checkpoint
from utils import (
    SegPyError,
    ensure_dir,
    image_to_float_array,
    pad_array_to_stride,
    read_rgb_image,
    receptive_field_to_stride,
    require_torch,
    setup_logging,
    write_json_result,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict binary foreground masks with a trained SVGF16 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--log-level", default="INFO")
    return parser


def iter_images(image: str | None, image_dir: str | None) -> list[Path]:
    if image and image_dir:
        raise SegPyError("Use only one of --image or --image-dir.")
    if not image and not image_dir:
        raise SegPyError("One of --image or --image-dir is required.")
    if image:
        path = Path(image)
        if not path.exists():
            raise SegPyError(f"Input image does not exist: {path}")
        return [path]
    root = Path(image_dir or "")
    if not root.exists() or not root.is_dir():
        raise SegPyError(f"image-dir does not exist or is not a directory: {root}")
    images = [
        item
        for item in sorted(root.iterdir())
        if item.is_file() and item.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}
    ]
    if not images:
        raise SegPyError(f"No images found in image-dir: {root}")
    return images


def render_overlay(image: Image.Image, mask: np.ndarray, alpha: float) -> Image.Image:
    if not 0.0 <= alpha <= 1.0:
        raise SegPyError(f"alpha must be in [0,1], got {alpha}")
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    overlay[mask > 0] = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    rendered = np.where(mask[..., None] > 0, base * (1.0 - alpha) + overlay * alpha, base)
    return Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8), mode="RGB")


def predict(args: argparse.Namespace) -> dict[str, object]:
    torch = require_torch()
    logger = logging.getLogger("segpy.predict")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        device = "cpu"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SegPyError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_svgf16_from_checkpoint(checkpoint, device=device)
    config = checkpoint.get("model_config", {})
    receptive_field = int(config.get("receptive_field", 64))
    stride = receptive_field_to_stride(receptive_field)

    output_dir = ensure_dir(args.output_dir)
    results = []
    for image_path in iter_images(args.image, args.image_dir):
        image = read_rgb_image(image_path)
        array = image_to_float_array(image)
        padded, original_size = pad_array_to_stride(array, stride=stride, mode="edge")
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(tensor)
            prediction = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        original_h, original_w = original_size
        prediction = prediction[:original_h, :original_w]
        binary = (prediction > 0).astype(np.uint8) * 255
        if int(binary.sum()) == 0:
            logger.info("No foreground pixels detected for image: %s", image_path)

        mask_path = output_dir / f"{image_path.stem}_mask.png"
        render_path = output_dir / f"{image_path.stem}_render.png"
        Image.fromarray(binary, mode="L").save(mask_path)
        render_overlay(image, binary, args.alpha).save(render_path)
        results.append({"image": str(image_path), "mask": str(mask_path), "render": str(render_path)})

    return {"ok": True, "count": len(results), "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        write_json_result(predict(args))
        return 0
    except Exception as exc:
        logging.getLogger("segpy.predict").exception("Prediction failed")
        write_json_result({"ok": False, "errorMessage": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

