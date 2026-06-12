from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from utils import SegPyError


@dataclass(frozen=True)
class YOLOPolygon:
    class_id: int
    points: np.ndarray
    line_number: int


@dataclass(frozen=True)
class YOLOAnnotation:
    path: Path
    polygons: list[YOLOPolygon]


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg"}


def parse_yolo_segmentation_label(
    label_path: str | Path,
    image_size: tuple[int, int],
    num_foreground_classes: int | None = None,
    allow_empty: bool = True,
) -> YOLOAnnotation:
    path = Path(label_path)
    if not path.exists():
        raise SegPyError(f"YOLO label file does not exist: {path}")
    if not path.is_file():
        raise SegPyError(f"YOLO label path is not a file: {path}")

    width, height = image_size
    if width <= 0 or height <= 0:
        raise SegPyError(f"Invalid image size for YOLO label parsing: width={width}, height={height}")

    polygons: list[YOLOPolygon] = []
    text = path.read_text(encoding="utf-8-sig")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 7 or len(fields) % 2 == 0:
            raise SegPyError(
                f"Invalid YOLO segmentation line at {path}:{line_number}. "
                "Expected: class x1 y1 x2 y2 x3 y3 ..."
            )
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise SegPyError(f"Invalid YOLO class id at {path}:{line_number}: {fields[0]!r}") from exc
        if class_id < 0:
            raise SegPyError(f"YOLO class id must be >= 0 at {path}:{line_number}, got {class_id}")
        if num_foreground_classes is not None and class_id >= num_foreground_classes:
            raise SegPyError(
                f"YOLO class id out of range at {path}:{line_number}: "
                f"class={class_id}, numClasses={num_foreground_classes}"
            )

        coords: list[float] = []
        for value in fields[1:]:
            try:
                coord = float(value)
            except ValueError as exc:
                raise SegPyError(f"Invalid YOLO coordinate at {path}:{line_number}: {value!r}") from exc
            if not np.isfinite(coord) or coord < 0.0 or coord > 1.0:
                raise SegPyError(
                    f"YOLO coordinate must be normalized to [0,1] at {path}:{line_number}, got {coord}"
                )
            coords.append(coord)

        points = []
        for index in range(0, len(coords), 2):
            points.append((coords[index] * width, coords[index + 1] * height))
        if len(points) < 3:
            raise SegPyError(f"YOLO polygon must contain at least 3 points at {path}:{line_number}")
        polygons.append(
            YOLOPolygon(
                class_id=class_id + 1,
                points=np.asarray(points, dtype=np.float32),
                line_number=line_number,
            )
        )

    if not polygons and not allow_empty:
        raise SegPyError(f"YOLO label file contains no polygons: {path}")
    return YOLOAnnotation(path=path, polygons=polygons)


def yolo_annotation_to_mask(annotation: YOLOAnnotation, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon in annotation.polygons:
        points = [(float(x), float(y)) for x, y in polygon.points]
        draw.polygon(points, fill=int(polygon.class_id))
    return np.asarray(mask_image, dtype=np.uint8)


def parse_yolo_names(raw_names: Any) -> list[str] | None:
    if raw_names is None:
        return None
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    if isinstance(raw_names, dict):
        items = sorted((int(key), str(value)) for key, value in raw_names.items())
        if not items:
            return []
        max_id = items[-1][0]
        names = [f"class_{index}" for index in range(max_id + 1)]
        for index, value in items:
            if index < 0:
                raise SegPyError(f"YOLO names contains a negative class id: {index}")
            names[index] = value
        return names
    raise SegPyError("YOLO dataset.yaml names must be a list or dict")


def load_yolo_dataset_yaml(dataset_yaml: str | Path, split: str) -> tuple[Path, Path, list[str] | None]:
    path = Path(dataset_yaml)
    if not path.exists():
        raise SegPyError(f"YOLO dataset.yaml does not exist: {path}")
    if not path.is_file():
        raise SegPyError(f"YOLO dataset.yaml path is not a file: {path}")
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise SegPyError("PyYAML is required to read YOLO dataset.yaml. Install pyyaml.") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise SegPyError(f"YOLO dataset.yaml must contain a mapping: {path}")
    if split not in data:
        raise SegPyError(f"YOLO dataset.yaml does not contain split {split!r}: {path}")

    root_value = data.get("path", path.parent)
    root_path = Path(root_value)
    if not root_path.is_absolute():
        root_path = (path.parent / root_path).resolve()

    split_value = data[split]
    if isinstance(split_value, list):
        raise SegPyError("YOLO dataset.yaml list splits are not supported; use a directory path.")
    image_dir = Path(str(split_value))
    if not image_dir.is_absolute():
        image_dir = (root_path / image_dir).resolve()
    if not image_dir.exists() or not image_dir.is_dir():
        raise SegPyError(f"YOLO image directory from dataset.yaml does not exist: {image_dir}")

    parts = list(image_dir.parts)
    if "images" in parts:
        index = parts.index("images")
        label_dir = Path(*parts[:index], "labels", *parts[index + 1 :])
    else:
        label_dir = image_dir.parent / "labels" / image_dir.name
    if not label_dir.exists() or not label_dir.is_dir():
        raise SegPyError(f"YOLO label directory inferred from dataset.yaml does not exist: {label_dir}")

    return image_dir, label_dir, parse_yolo_names(data.get("names"))


def find_yolo_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise SegPyError(f"YOLO image directory does not exist: {root}")
    if not root.is_dir():
        raise SegPyError(f"YOLO image path is not a directory: {root}")
    images = sorted(item for item in root.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SegPyError(f"No YOLO images found in directory: {root}")
    return images
