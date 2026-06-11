from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from utils import SegPyError


LOGGER = logging.getLogger("segpy.gt3")


@dataclass(frozen=True)
class Gt3Contour:
    points: np.ndarray
    byte_offset: int

    @property
    def area(self) -> float:
        x = self.points[:, 0]
        y = self.points[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        x = self.points[:, 0]
        y = self.points[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())


@dataclass(frozen=True)
class Gt3Annotation:
    path: Path
    expected_count: int | None
    contours: list[Gt3Contour]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _valid_xy(x: float, y: float, width: int, height: int) -> bool:
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    return -0.5 <= x <= width + 0.5 and -0.5 <= y <= height + 0.5


def parse_gt3(
    gt3_path: str | Path,
    image_size: tuple[int, int],
    min_points: int = 3,
    min_area: float = 1.0,
) -> Gt3Annotation:
    """Parse TeAiFlow .gt3 annotation by extracting valid float32 contour runs.

    The observed .gt3 files store polygon vertices as contiguous little-endian
    float32 x/y pairs in source-image coordinates, separated by binary metadata.
    This parser extracts maximal valid x/y runs and rejects tiny/no-area runs.
    """

    path = Path(gt3_path)
    if not path.exists():
        raise SegPyError(f".gt3 file does not exist: {path}")
    if not path.is_file():
        raise SegPyError(f".gt3 path is not a file: {path}")

    data = path.read_bytes()
    if len(data) < 32:
        raise SegPyError(f".gt3 file is too small to contain annotations: {path}")

    width, height = image_size
    if width <= 0 or height <= 0:
        raise SegPyError(f"Invalid image size for .gt3 parsing: width={width}, height={height}")

    expected_count = None
    try:
        count_a = _u32(data, 8)
        count_b = _u32(data, 12)
        if count_a == count_b and 0 < count_a < 10000:
            expected_count = int(count_a)
    except struct.error:
        expected_count = None

    contours: list[Gt3Contour] = []
    offset = 0
    while offset <= len(data) - 8:
        try:
            x = _f32(data, offset)
            y = _f32(data, offset + 4)
        except struct.error:
            break

        if not _valid_xy(x, y, width, height):
            offset += 4
            continue

        points: list[tuple[float, float]] = []
        cursor = offset
        while cursor <= len(data) - 8:
            x = _f32(data, cursor)
            y = _f32(data, cursor + 4)
            if not _valid_xy(x, y, width, height):
                break
            points.append((x, y))
            cursor += 8

        if len(points) >= min_points:
            array = np.asarray(points, dtype=np.float32)
            contour = Gt3Contour(points=array, byte_offset=offset)
            min_x, min_y, max_x, max_y = contour.bounds
            if contour.area >= min_area and (max_x - min_x) >= 1.0 and (max_y - min_y) >= 1.0:
                contours.append(contour)
                offset = cursor
                continue

        offset += 4

    if not contours:
        raise SegPyError(
            f"No valid foreground contours were parsed from .gt3 file: {path}. "
            f"imageSize={width}x{height}, bytes={len(data)}"
        )

    if expected_count is not None and expected_count != len(contours):
        LOGGER.warning(
            "Parsed contour count differs from .gt3 header for %s: header=%s parsed=%s",
            path,
            expected_count,
            len(contours),
        )

    return Gt3Annotation(path=path, expected_count=expected_count, contours=contours)


def annotation_to_mask(annotation: Gt3Annotation, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for contour in annotation.contours:
        points = [(float(x), float(y)) for x, y in contour.points]
        draw.polygon(points, fill=1)
    mask = np.asarray(mask_image, dtype=np.uint8)
    if int(mask.sum()) <= 0:
        raise SegPyError(f"Parsed contours produced an empty mask: {annotation.path}")
    return mask

