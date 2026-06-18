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
    receptive_field_to_patch_size,
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
    parser.add_argument("--image-dir", "--folder", dest="image_dir", default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--save-txt", "--save_txt", dest="save_txt", type=str2bool, default=True)
    parser.add_argument("--show", type=str2bool, default=False)
    parser.add_argument("--max-det", "--max_det", dest="max_det", type=int, default=1000)
    parser.add_argument("--min-pixel", "--min_pixel", "--min-pxiel", dest="min_pixel", type=int, default=1)
    parser.add_argument("--mask-thresh", "--mask_thresh", dest="mask_thresh", type=float, default=0.0)
    parser.add_argument("--tile", action="store_true", help="Enable tiled inference for very large images.")
    parser.add_argument("--tile-size", "--tile_size", dest="tile_size", type=int, default=None)
    parser.add_argument("--tile-overlap", "--tile_overlap", dest="tile_overlap", type=int, default=None)
    parser.add_argument("--no-tile", "--no_tile", dest="no_tile", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def iter_images(image: str | None, image_dir: str | None) -> list[Path]:
    if image and image_dir:
        raise SegPyError("Use only one of --image, --image-dir, or --folder.")
    if not image and not image_dir:
        raise SegPyError("One of --image, --image-dir, or --folder is required.")
    if image:
        path = Path(image)
        if not path.exists():
            raise SegPyError(f"Input image does not exist: {path}")
        return [path]
    root = Path(image_dir or "")
    if not root.exists() or not root.is_dir():
        raise SegPyError(f"image-dir/folder does not exist or is not a directory: {root}")
    images = [
        item
        for item in sorted(root.iterdir())
        if item.is_file() and item.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ]
    if not images:
        raise SegPyError(f"No images found in image-dir/folder: {root}")
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


def _tile_starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= 0:
        raise SegPyError(f"Invalid image dimension for tiling: {length}")
    if tile_size <= 0:
        raise SegPyError(f"tile_size must be > 0, got {tile_size}")
    if step <= 0:
        raise SegPyError(f"tile step must be > 0, got {step}")
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size, 0) + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _resolve_tile_settings(
    checkpoint: dict[str, object],
    receptive_field: int,
    stride: int,
    args: argparse.Namespace,
) -> tuple[int, int, bool]:
    preprocessing = checkpoint.get("preprocessing", {})
    checkpoint_patch_size = None
    if isinstance(preprocessing, dict):
        raw_patch_size = preprocessing.get("patch_size")
        if raw_patch_size is not None:
            checkpoint_patch_size = int(raw_patch_size)
    tile_size = int(args.tile_size or checkpoint_patch_size or receptive_field_to_patch_size(receptive_field))
    if tile_size <= 0:
        raise SegPyError(f"tile-size must be > 0, got {tile_size}")
    overlap = int(args.tile_overlap if args.tile_overlap is not None else max(stride, receptive_field))
    if overlap < 0:
        raise SegPyError(f"tile-overlap must be >= 0, got {overlap}")
    if overlap >= tile_size:
        raise SegPyError(f"tile-overlap must be smaller than tile-size: overlap={overlap}, tileSize={tile_size}")
    no_tile = bool(args.no_tile or not args.tile)
    return tile_size, overlap, no_tile


def _infer_logits_whole_image(
    torch: object,
    model: object,
    array: np.ndarray,
    stride: int,
    device: str,
) -> np.ndarray:
    padded, original_size = pad_array_to_stride(array, stride=stride, mode="edge")
    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits_tensor = model(tensor).squeeze(0).detach().cpu()
    original_h, original_w = original_size
    return logits_tensor.numpy()[:, :original_h, :original_w]


def _infer_logits_tiled(
    torch: object,
    model: object,
    array: np.ndarray,
    num_classes: int,
    stride: int,
    device: str,
    tile_size: int,
    tile_overlap: int,
    logger: logging.Logger,
) -> np.ndarray:
    height, width = array.shape[:2]
    step = tile_size - tile_overlap
    y_starts = _tile_starts(height, tile_size, step)
    x_starts = _tile_starts(width, tile_size, step)
    tile_count = len(y_starts) * len(x_starts)
    logger.info(
        "Using tiled inference: image=%sx%s tile_size=%s overlap=%s stride=%s tiles=%s",
        width,
        height,
        tile_size,
        tile_overlap,
        stride,
        tile_count,
    )

    logits_sum = np.zeros((num_classes, height, width), dtype=np.float32)
    logits_count = np.zeros((height, width), dtype=np.float32)
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            tile = array[y0:y1, x0:x1, :]
            tile_logits = _infer_logits_whole_image(torch, model, tile, stride, device)
            if tile_logits.shape != (num_classes, y1 - y0, x1 - x0):
                raise SegPyError(
                    f"Invalid tile logits shape: got={tile_logits.shape}, "
                    f"expected={(num_classes, y1 - y0, x1 - x0)}, tile=({x0},{y0},{x1},{y1})"
                )
            logits_sum[:, y0:y1, x0:x1] += tile_logits
            logits_count[y0:y1, x0:x1] += 1.0

    if np.any(logits_count <= 0):
        raise SegPyError("Tiled inference failed: some pixels were not covered by any tile.")
    return logits_sum / logits_count[None, :, :]


def _infer_logits(
    torch: object,
    model: object,
    array: np.ndarray,
    num_classes: int,
    stride: int,
    device: str,
    tile_size: int,
    tile_overlap: int,
    no_tile: bool,
    logger: logging.Logger,
) -> tuple[np.ndarray, str]:
    height, width = array.shape[:2]
    if no_tile or (height <= tile_size and width <= tile_size):
        return _infer_logits_whole_image(torch, model, array, stride, device), "whole"
    return (
        _infer_logits_tiled(
            torch=torch,
            model=model,
            array=array,
            num_classes=num_classes,
            stride=stride,
            device=device,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            logger=logger,
        ),
        "tiled",
    )


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
    tile_size, tile_overlap, no_tile = _resolve_tile_settings(checkpoint, receptive_field, stride, args)

    output_dir = ensure_dir(args.output_dir)
    results = []
    for image_path in iter_images(args.image, args.image_dir):
        image = read_rgb_image(image_path)
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

        instance_result = postprocess_logits(
            logits=logits,
            classes=classes,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            min_pixel=args.min_pixel,
            mask_thresh=args.mask_thresh,
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
            "inference": {
                "mode": inference_mode,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "stride": stride,
                "min_pixel": args.min_pixel,
                "mask_thresh": args.mask_thresh,
            },
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
                "inferenceMode": inference_mode,
                "instances": instance_result.json_instances(),
                "label": str(label_path),
                "mask": str(mask_path),
                "instanceMask": str(instances_path),
                "render": str(render_path),
                "json": str(json_path),
                "txt": str(txt_path) if args.save_txt else None,
            }
        )

    # return {"ok": True, "count": len(results), "results": results}
    return {"ok": True, "count": len(results), "results num": len(instance_result.instances)}



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
