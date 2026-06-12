from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from instance_postprocess import postprocess_logits, render_instances, write_instances_txt
from seg_models import build_svgf16_from_checkpoint
from utils import (
    SegPyError,
    ensure_dir,
    image_to_float_array,
    load_torch_checkpoint,
    pad_array_to_stride,
    read_rgb_image,
    receptive_field_to_stride,
    require_torch,
    setup_logging,
    str2bool,
    write_json_result,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict YOLO-seg style instances with a trained SVGF16 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--save-txt", "--save_txt", dest="save_txt", type=str2bool, default=True)
    parser.add_argument("--show", type=str2bool, default=False)
    parser.add_argument("--max-det", "--max_det", dest="max_det", type=int, default=1000)
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


def _classes_from_checkpoint(checkpoint: dict[str, object], num_classes: int) -> list[str]:
    raw_classes = checkpoint.get("classes")
    if isinstance(raw_classes, list):
        classes = [str(item) for item in raw_classes]
    else:
        classes = []
    while len(classes) < num_classes:
        classes.append(f"class_{len(classes)}")
    return classes[:num_classes]


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
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    model = build_svgf16_from_checkpoint(checkpoint, device=device)
    config = checkpoint.get("model_config", {})
    if not isinstance(config, dict):
        raise SegPyError("Checkpoint model_config must be a dict.")
    receptive_field = int(config.get("receptive_field", 64))
    num_classes = int(config.get("num_classes", 2))
    classes = _classes_from_checkpoint(checkpoint, num_classes)
    stride = receptive_field_to_stride(receptive_field)

    output_dir = ensure_dir(args.output_dir)
    results = []
    for image_path in iter_images(args.image, args.image_dir):
        image = read_rgb_image(image_path)
        array = image_to_float_array(image)
        padded, original_size = pad_array_to_stride(array, stride=stride, mode="edge")
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits_tensor = model(tensor).squeeze(0).detach().cpu()
        original_h, original_w = original_size
        logits = logits_tensor.numpy()[:, :original_h, :original_w]

        instance_result = postprocess_logits(
            logits=logits,
            classes=classes,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
        )
        if len(instance_result.instances) == 0:
            logger.info("No foreground instances detected for image: %s", image_path)

        label_path = output_dir / f"{image_path.stem}_label.png"
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        instances_path = output_dir / f"{image_path.stem}_instances.png"
        render_path = output_dir / f"{image_path.stem}_render.png"
        json_path = output_dir / f"{image_path.stem}_results.json"
        txt_path = output_dir / f"{image_path.stem}_results.txt"

        Image.fromarray(instance_result.label_map.astype(np.uint8), mode="L").save(label_path)
        Image.fromarray(instance_result.binary_mask.astype(np.uint8), mode="L").save(mask_path)
        Image.fromarray(instance_result.instance_mask.astype(np.uint16), mode="I;16").save(instances_path)
        render_image = render_instances(image, instance_result, alpha=args.alpha)
        render_image.save(render_path)

        json_payload = {
            "image": str(image_path),
            "width": int(image.width),
            "height": int(image.height),
            "classes": classes,
            "count": len(instance_result.instances),
            "instances": instance_result.json_instances(),
            "files": {
                "label": str(label_path),
                "mask": str(mask_path),
                "instances": str(instances_path),
                "render": str(render_path),
                "txt": str(txt_path) if args.save_txt else None,
            },
        }
        json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.save_txt:
            write_instances_txt(str(txt_path), instance_result.instances)
        if args.show:
            render_image.show()

        results.append(
            {
                "image": str(image_path),
                "count": len(instance_result.instances),
                "instances": instance_result.json_instances(),
                "label": str(label_path),
                "mask": str(mask_path),
                "instanceMask": str(instances_path),
                "render": str(render_path),
                "json": str(json_path),
                "txt": str(txt_path) if args.save_txt else None,
            }
        )

    return {"ok": True, "count": len(results), "results": results}


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_arg_parser()
        args = parser.parse_args(argv)
        setup_logging(args.log_level)
        write_json_result(predict(args))
        return 0
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        write_json_result(
            {"ok": False, "errorMessage": f"Invalid command line arguments. Exit code: {exc.code}"}
        )
        return int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        logging.getLogger("segpy.predict").exception("Prediction failed")
        write_json_result({"ok": False, "errorMessage": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
