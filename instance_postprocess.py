from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from utils import SegPyError, require_opencv, stable_color


@dataclass
class InstancePrediction:
    id: int
    cls: int
    class_id: int
    class_name: str
    score: float
    polygon: list[list[int]]
    bbox: list[int]
    area: int
    _mask: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cls": self.cls,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": round(float(self.score), 2),
            "polygon": self.polygon,
            "bbox": self.bbox,
            "area": int(self.area),
        }


@dataclass
class InstanceSegResult:
    label_map: np.ndarray
    binary_mask: np.ndarray
    instance_mask: np.ndarray
    instances: list[InstancePrediction]

    def json_instances(self) -> list[dict[str, Any]]:
        return [item.to_json() for item in self.instances]


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 3:
        raise SegPyError(f"logits must have shape [C,H,W], got {logits.shape}")
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=0, keepdims=True)
    if np.any(denom <= 0):
        raise SegPyError("Invalid softmax denominator; logits contain non-finite values.")
    return exp / denom


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    if intersection <= 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return 0.0 if union <= 0 else float(intersection) / float(union)


def _instance_mask_iou(a: InstancePrediction, b: InstancePrediction) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    a_crop = a._mask[iy1 - ay1 : iy2 - ay1, ix1 - ax1 : ix2 - ax1]
    b_crop = b._mask[iy1 - by1 : iy2 - by1, ix1 - bx1 : ix2 - bx1]
    intersection = int(np.logical_and(a_crop, b_crop).sum())
    if intersection <= 0:
        return 0.0
    union = int(a.area + b.area - intersection)
    return 0.0 if union <= 0 else float(intersection) / float(union)


def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise SegPyError("Cannot compute bbox for empty mask.")
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2, y2]


def _polygon_from_mask(mask: np.ndarray) -> list[list[int]]:
    cv2 = require_opencv()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2)
    if len(points) >= 3:
        return [[int(x), int(y)] for x, y in points]
    x1, y1, x2, y2 = _bbox_from_mask(mask > 0)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _normalize_classes(classes: list[str] | None, num_classes: int) -> list[str]:
    if classes is None:
        classes = []
    names = [str(item) for item in classes]
    while len(names) < num_classes:
        names.append(f"class_{len(names)}")
    return names[:num_classes]


def postprocess_logits(
    logits: np.ndarray,
    classes: list[str] | None,
    conf: float = 0.2,
    iou: float = 0.5,
    max_det: int = 1000,
    min_pixel: int = 1,
    mask_thresh: float = 0.0,
) -> InstanceSegResult:
    if not 0.0 <= conf <= 1.0:
        raise SegPyError(f"conf must be in [0,1], got {conf}")
    if not 0.0 <= iou <= 1.0:
        raise SegPyError(f"iou must be in [0,1], got {iou}")
    if not 0.0 <= mask_thresh <= 1.0:
        raise SegPyError(f"mask_thresh must be in [0,1], got {mask_thresh}")
    if max_det <= 0 or max_det > 65535:
        raise SegPyError(f"max_det must be in [1,65535], got {max_det}")
    if min_pixel <= 0:
        raise SegPyError(f"min_pixel must be > 0, got {min_pixel}")
    if not np.all(np.isfinite(logits)):
        raise SegPyError("logits contain NaN or Inf values.")

    probs = _softmax(logits.astype(np.float32, copy=False))
    raw_label_map = np.argmax(probs, axis=0).astype(np.uint8)
    raw_score_map = np.take_along_axis(probs, raw_label_map[None, :, :], axis=0)[0]
    label_map = np.where(
        (raw_label_map > 0) & (raw_score_map >= mask_thresh),
        raw_label_map,
        0,
    ).astype(np.uint8)
    num_classes = int(logits.shape[0])
    class_names = _normalize_classes(classes, num_classes)
    height, width = label_map.shape

    cv2 = require_opencv()
    candidates: list[InstancePrediction] = []
    for class_id in range(1, num_classes):
        class_mask = (label_map == class_id).astype(np.uint8)
        if int(class_mask.sum()) <= 0:
            continue
        component_count, component_labels = cv2.connectedComponents(class_mask, connectivity=8)
        for component_id in range(1, component_count):
            component_mask = component_labels == component_id
            area = int(component_mask.sum())
            if area < min_pixel:
                continue
            score = float(probs[class_id][component_mask].mean())
            if score < conf:
                continue
            bbox = _bbox_from_mask(component_mask)
            x1, y1, x2, y2 = bbox
            component_crop = component_mask[y1:y2, x1:x2].copy()
            polygon = _polygon_from_mask(component_crop)
            if not polygon:
                continue
            polygon = [[int(x + x1), int(y + y1)] for x, y in polygon]
            candidates.append(
                InstancePrediction(
                    id=0,
                    cls=class_id - 1,
                    class_id=class_id,
                    class_name=class_names[class_id],
                    score=round(score, 2),
                    polygon=polygon,
                    bbox=bbox,
                    area=area,
                    _mask=component_crop,
                )
            )

    candidates.sort(key=lambda item: item.score, reverse=True)
    selected: list[InstancePrediction] = []
    for candidate in candidates:
        suppressed = False
        for existing in selected:
            if candidate.class_id == existing.class_id and _instance_mask_iou(candidate, existing) > iou:
                suppressed = True
                break
        if suppressed:
            continue
        candidate.id = len(selected) + 1
        selected.append(candidate)
        if len(selected) >= max_det:
            break

    filtered_label_map = np.zeros((height, width), dtype=np.uint8)
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    for item in selected:
        x1, y1, x2, y2 = item.bbox
        instance_target = instance_mask[y1:y2, x1:x2]
        instance_target[item._mask] = int(item.id)
        label_target = filtered_label_map[y1:y2, x1:x2]
        label_target[item._mask] = int(item.class_id)
    binary_mask = (filtered_label_map > 0).astype(np.uint8) * 255

    return InstanceSegResult(
        label_map=filtered_label_map,
        binary_mask=binary_mask,
        instance_mask=instance_mask,
        instances=selected,
    )


def render_instances(
    image: Image.Image,
    result: InstanceSegResult,
    alpha: float = 0.45,
) -> Image.Image:
    if not 0.0 <= alpha <= 1.0:
        raise SegPyError(f"alpha must be in [0,1], got {alpha}")
    cv2 = require_opencv()
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    rendered = base.copy()

    for instance in result.instances:
        color = np.asarray(stable_color(instance.class_id), dtype=np.float32)
        mask = result.instance_mask == instance.id
        rendered[mask] = rendered[mask] * (1.0 - alpha) + color * alpha

    rendered_u8 = np.clip(rendered, 0, 255).astype(np.uint8)
    for instance in result.instances:
        color = stable_color(instance.class_id)
        polygon = np.asarray(instance.polygon, dtype=np.int32)
        if polygon.size:
            cv2.polylines(rendered_u8, [polygon], isClosed=True, color=color, thickness=2)
        x1, y1, x2, y2 = instance.bbox
        cv2.rectangle(rendered_u8, (x1, y1), (x2, y2), color, 2)
        label = f"{instance.class_name} {instance.score:.2f}"
        text_x = max(0, x1)
        text_y = max(12, y1 - 4)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            rendered_u8,
            (text_x, text_y - text_h - baseline - 2),
            (text_x + text_w + 4, text_y + baseline),
            (255, 255, 255),
            thickness=-1,
        )
        cv2.putText(
            rendered_u8,
            label,
            (text_x + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return Image.fromarray(rendered_u8, mode="RGB")


def write_instances_txt(path: str, instances: list[InstancePrediction]) -> None:
    lines = []
    for item in instances:
        coords: list[str] = []
        for x, y in item.polygon:
            coords.append(str(int(x)))
            coords.append(str(int(y)))
        lines.append(" ".join([str(int(item.cls)), f"{item.score:.2f}", *coords]))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        if lines:
            handle.write("\n")
