from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from utils import SegPyError


LOGGER = logging.getLogger("segpy.gt3")
_COUNT_MISMATCH_WARNED: set[Path] = set()


@dataclass(frozen=True)
class Gt3Contour:
    points: np.ndarray
    byte_offset: int
    class_id: int
    raw_gt_value: int

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


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


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
    for offset in range(0, len(data) - 64, 4):
        try:
            point_count = _u32(data, offset)
            point_count_repeat = _u32(data, offset + 4)
        except struct.error:
            continue
        if point_count != point_count_repeat:
            continue
        if point_count < min_points or point_count > 10000:
            continue

        points_offset = offset + 56
        points_end = points_offset + int(point_count) * 8
        if points_end > len(data):
            continue

        points: list[tuple[float, float]] = []
        valid = True
        for point_index in range(int(point_count)):
            point_offset = points_offset + point_index * 8
            x = _f32(data, point_offset)
            y = _f32(data, point_offset + 4)
            if not _valid_xy(x, y, width, height):
                valid = False
                break
            points.append((x, y))
        if not valid:
            continue

        array = np.asarray(points, dtype=np.float32)
        min_x = float(array[:, 0].min())
        min_y = float(array[:, 1].min())
        max_x = float(array[:, 0].max())
        max_y = float(array[:, 1].max())
        area = _polygon_area(array)
        if area < min_area or (max_x - min_x) < 1.0 or (max_y - min_y) < 1.0:
            continue

        if offset < 24:
            continue
        raw_gt_value = _u32(data, offset - 24)
        class_id = int(raw_gt_value) - 2
        if class_id <= 0:
            LOGGER.info(
                "Skip non-foreground .gt3 contour: path=%s offset=%s rawGtValue=%s",
                path,
                offset,
                raw_gt_value,
            )
            continue
        contours.append(
            Gt3Contour(
                points=array,
                byte_offset=offset,
                class_id=class_id,
                raw_gt_value=int(raw_gt_value),
            )
        )

    if not contours:
        raise SegPyError(
            f"No valid foreground contours were parsed from .gt3 file: {path}. "
            f"imageSize={width}x{height}, bytes={len(data)}"
        )

    resolved_path = path.resolve()
    if (
        expected_count is not None
        and expected_count != len(contours)
        and resolved_path not in _COUNT_MISMATCH_WARNED
    ):
        LOGGER.warning(
            "Parsed contour count differs from .gt3 header for %s: header=%s parsed=%s",
            path,
            expected_count,
            len(contours),
        )
        _COUNT_MISMATCH_WARNED.add(resolved_path)

    return Gt3Annotation(path=path, expected_count=expected_count, contours=contours)


def annotation_to_mask(annotation: Gt3Annotation, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for contour in annotation.contours:
        points = [(float(x), float(y)) for x, y in contour.points]
        if contour.class_id <= 0:
            continue
        draw.polygon(points, fill=int(contour.class_id))
    mask = np.asarray(mask_image, dtype=np.uint8)
    if int(mask.sum()) <= 0:
        raise SegPyError(f"Parsed contours produced an empty mask: {annotation.path}")
    return mask
