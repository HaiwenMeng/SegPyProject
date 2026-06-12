from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LOGGER_NAME = "segpy"


class SegPyError(RuntimeError):
    """Base error with a user-facing errorMessage."""


class DependencyError(SegPyError):
    """Raised when an optional runtime dependency is missing."""


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    return logger


def require_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise DependencyError(
            "PyTorch is required but is not installed. Install torch before running this command."
        ) from exc
    return torch


def require_opencv():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise DependencyError(
            "OpenCV is required for instance postprocessing. Install opencv-python before running this command."
        ) from exc
    return cv2


def load_torch_checkpoint(checkpoint_path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    torch = require_torch()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise SegPyError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}: {checkpoint_path}")
    return checkpoint


def str2bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise SegPyError(f"Invalid boolean value: {value!r}")


def stable_color(index: int) -> tuple[int, int, int]:
    palette = [
        (0, 255, 0),
        (255, 64, 64),
        (64, 160, 255),
        (255, 192, 0),
        (192, 64, 255),
        (0, 220, 220),
        (255, 96, 192),
        (128, 255, 96),
    ]
    if index < len(palette):
        return palette[index]
    return (
        int((37 * index + 53) % 200 + 40),
        int((83 * index + 97) % 200 + 40),
        int((131 * index + 17) % 200 + 40),
    )


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_rgb_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise SegPyError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise SegPyError(f"Image path is not a file: {image_path}")
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise SegPyError(f"Failed to read image file: {image_path}") from exc


def image_to_float_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def pad_array_to_stride(
    array: np.ndarray,
    stride: int,
    mode: str,
    constant_values: int | float = 0,
) -> tuple[np.ndarray, tuple[int, int]]:
    if stride <= 0:
        raise SegPyError(f"Invalid stride: {stride}")
    height, width = array.shape[:2]
    pad_h = (stride - height % stride) % stride
    pad_w = (stride - width % stride) % stride
    if pad_h == 0 and pad_w == 0:
        return array, (height, width)
    pad_spec = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (array.ndim - 2)
    if mode == "constant":
        padded = np.pad(array, pad_spec, mode=mode, constant_values=constant_values)
    else:
        padded = np.pad(array, pad_spec, mode=mode)
    return padded, (height, width)


def write_json_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def checkpoint_default_path(project_root: str | Path, receptive_field: int) -> Path:
    return Path(project_root) / "outputs" / "checkpoints" / f"svgf16_rf{receptive_field}_best.pt"


def receptive_field_to_patch_size(receptive_field: int) -> int:
    mapping = {32: 256, 64: 512, 128: 768, 256: 1024}
    if receptive_field not in mapping:
        raise SegPyError(
            f"Unsupported receptive_field={receptive_field}. Supported values: {sorted(mapping)}"
        )
    return mapping[receptive_field]


def receptive_field_to_stride(receptive_field: int) -> int:
    mapping = {32: 8, 64: 16, 128: 32, 256: 64}
    if receptive_field not in mapping:
        raise SegPyError(
            f"Unsupported receptive_field={receptive_field}. Supported values: {sorted(mapping)}"
        )
    return mapping[receptive_field]
