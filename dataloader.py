from __future__ import annotations

import logging
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from gt3_parser import annotation_to_mask, parse_gt3
from utils import DependencyError, SegPyError, image_to_float_array, read_rgb_image, receptive_field_to_patch_size
from yolo_parser import (
    find_yolo_images,
    load_yolo_dataset_yaml,
    parse_yolo_segmentation_label,
    yolo_annotation_to_mask,
)

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - depends on runtime env
    torch = None  # type: ignore
    Dataset = object  # type: ignore


LOGGER = logging.getLogger("segpy.dataloader")


@dataclass(frozen=True)
class SegSample:
    label_path: Path
    image_path: Path
    name: str


def _require_torch() -> None:
    if torch is None:
        raise DependencyError(
            "PyTorch is required but is not installed. Install torch before using segmentation dataloaders."
        )


def _read_te_classes(gt_dir: Path) -> list[str] | None:
    db_paths = sorted(gt_dir.glob("*_Inq.db"))
    if not db_paths:
        LOGGER.warning("No *_Inq.db found in .gt3 directory; class names will be inferred from masks: %s", gt_dir)
        return None
    db_path = db_paths[0]
    try:
        con = sqlite3.connect(str(db_path))
        rows = con.execute("select GtID, GtName from GtNameTab where GtID >= 4 order by GtID").fetchall()
    except Exception as exc:
        raise SegPyError(f"Failed to read TeAiFlow class names from database: {db_path}") from exc
    finally:
        try:
            con.close()  # type: ignore[name-defined]
        except Exception:
            pass
    if not rows:
        LOGGER.warning("No GtID>=4 class names found in database; class names will be inferred: %s", db_path)
        return None
    return ["BG"] + [str(name) for _, name in rows]


def _classes_from_max_label(max_label: int) -> list[str]:
    if max_label < 0:
        raise SegPyError(f"Invalid max_label={max_label}")
    return ["BG"] + [f"class_{index}" for index in range(1, max_label + 1)]


class _PatchSegDataset(Dataset):  # type: ignore[misc]
    dataset_type = "base"

    def __init__(
        self,
        receptive_field: int = 64,
        patch_size: int | None = None,
        samples_per_image: int = 64,
        positive_ratio: float = 0.5,
        allow_empty_mask: bool = False,
        cache_samples: bool = True,
    ):
        _require_torch()
        self.receptive_field = int(receptive_field)
        self.patch_size = int(patch_size or receptive_field_to_patch_size(self.receptive_field))
        self.samples_per_image = int(samples_per_image)
        self.positive_ratio = float(positive_ratio)
        self.allow_empty_mask = bool(allow_empty_mask)
        self.cache_samples = bool(cache_samples)
        self._sample_cache: dict[Path, tuple[Image.Image, np.ndarray]] = {}
        self.samples: list[SegSample] = []
        self.classes: list[str] = ["BG", "defect"]
        self.max_label: int = 1
        self._label_range_ready = False

        if self.samples_per_image <= 0:
            raise SegPyError(f"samples_per_image must be > 0, got {samples_per_image}")
        if not 0.0 <= self.positive_ratio <= 1.0:
            raise SegPyError(f"positive_ratio must be in [0,1], got {positive_ratio}")
        if self.patch_size <= 0:
            raise SegPyError(f"patch_size must be > 0, got {self.patch_size}")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples) * self.samples_per_image

    def _load_mask(self, sample: SegSample, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    def load_full_sample(self, sample: SegSample) -> tuple[Image.Image, np.ndarray]:
        if self.cache_samples:
            cached = self._sample_cache.get(sample.label_path)
            if cached is not None:
                return cached
        image = read_rgb_image(sample.image_path)
        mask = self._load_mask(sample, image)
        if self._label_range_ready and int(mask.max()) > self.max_label:
            raise SegPyError(
                f"Mask label exceeds dataset class range for {sample.label_path}: "
                f"maxMask={int(mask.max())}, maxLabel={self.max_label}"
            )
        if int(mask.sum()) <= 0 and not self.allow_empty_mask:
            raise SegPyError(f"Mask is empty for sample: {sample.label_path}")
        if self.cache_samples:
            self._sample_cache[sample.label_path] = (image, mask)
        return image, mask

    def clear_cache(self) -> None:
        self._sample_cache.clear()

    @staticmethod
    def _pad_for_crop(image: np.ndarray, mask: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        height, width = mask.shape
        pad_h = max(0, patch_size - height)
        pad_w = max(0, patch_size - width)
        if pad_h == 0 and pad_w == 0:
            return image, mask
        image_pad = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode="edge")
        mask_pad = np.pad(mask, [(0, pad_h), (0, pad_w)], mode="constant", constant_values=0)
        return image_pad, mask_pad

    def _choose_crop(self, mask: np.ndarray) -> tuple[int, int]:
        height, width = mask.shape
        max_y = height - self.patch_size
        max_x = width - self.patch_size
        use_positive = random.random() < self.positive_ratio
        foreground = np.argwhere(mask > 0)
        if use_positive and foreground.size > 0:
            cy, cx = foreground[random.randrange(len(foreground))]
            y0 = int(np.clip(cy - self.patch_size // 2, 0, max_y))
            x0 = int(np.clip(cx - self.patch_size // 2, 0, max_x))
            return y0, x0
        y0 = random.randint(0, max_y) if max_y > 0 else 0
        x0 = random.randint(0, max_x) if max_x > 0 else 0
        return y0, x0

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % len(self.samples)]
        image, mask = self.load_full_sample(sample)
        image_array = image_to_float_array(image)
        image_array, mask = self._pad_for_crop(image_array, mask, self.patch_size)
        if int(mask.sum()) <= 0 and not self.allow_empty_mask:
            raise SegPyError(f"Mask is empty after preprocessing for sample: {sample.label_path}")
        y0, x0 = self._choose_crop(mask)
        y1 = y0 + self.patch_size
        x1 = x0 + self.patch_size
        image_patch = image_array[y0:y1, x0:x1, :]
        mask_patch = mask[y0:y1, x0:x1]
        if image_patch.shape[:2] != (self.patch_size, self.patch_size):
            raise SegPyError(
                f"Invalid image patch shape for {sample.name}: got={image_patch.shape}, patchSize={self.patch_size}"
            )
        if mask_patch.shape != (self.patch_size, self.patch_size):
            raise SegPyError(
                f"Invalid mask patch shape for {sample.name}: got={mask_patch.shape}, patchSize={self.patch_size}"
            )
        image_tensor = torch.from_numpy(image_patch.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_patch.astype(np.int64))
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_name": sample.name,
            "image_path": str(sample.image_path),
            "label_path": str(sample.label_path),
            "crop": torch.tensor([x0, y0, x1, y1], dtype=torch.int64),
        }


class TEDataloader(_PatchSegDataset):
    dataset_type = "gt3"

    def __init__(
        self,
        gt_dir: str | Path,
        image_dir: str | Path | None = None,
        receptive_field: int = 64,
        patch_size: int | None = None,
        samples_per_image: int = 64,
        positive_ratio: float = 0.5,
        validate_masks: bool = True,
        cache_samples: bool = True,
    ):
        super().__init__(
            receptive_field=receptive_field,
            patch_size=patch_size,
            samples_per_image=samples_per_image,
            positive_ratio=positive_ratio,
            allow_empty_mask=False,
            cache_samples=cache_samples,
        )
        self.gt_dir = Path(gt_dir)
        if not self.gt_dir.exists():
            raise SegPyError(f"gt_dir does not exist: {self.gt_dir}")
        if not self.gt_dir.is_dir():
            raise SegPyError(f"gt_dir is not a directory: {self.gt_dir}")

        self.image_dir = Path(image_dir) if image_dir is not None else self.gt_dir.parent / "1" / "SrcImage"
        if not self.image_dir.exists():
            raise SegPyError(f"image_dir does not exist: {self.image_dir}")
        if not self.image_dir.is_dir():
            raise SegPyError(f"image_dir is not a directory: {self.image_dir}")

        gt_paths = sorted(self.gt_dir.glob("*.gt3"))
        if not gt_paths:
            raise SegPyError(f"No .gt3 files found in gt_dir: {self.gt_dir}")

        for gt_path in gt_paths:
            image_path = self._match_image(gt_path.stem)
            self.samples.append(SegSample(label_path=gt_path, image_path=image_path, name=gt_path.stem))

        max_label = 0
        if validate_masks:
            for sample in self.samples:
                _, mask = self.load_full_sample(sample)
                max_label = max(max_label, int(mask.max()))
                if int(mask.sum()) <= 0:
                    raise SegPyError(f"Mask is empty for sample: {sample.label_path}")

        db_classes = _read_te_classes(self.gt_dir)
        self.max_label = max_label
        if db_classes is not None:
            if len(db_classes) <= self.max_label:
                raise SegPyError(
                    f"TeAiFlow class database does not contain enough classes: "
                    f"classes={db_classes}, maxMaskLabel={self.max_label}"
                )
            self.classes = db_classes
        else:
            self.classes = _classes_from_max_label(self.max_label)
        self._label_range_ready = True

        LOGGER.info(
            "TEDataloader initialized: gt_dir=%s image_dir=%s samples=%s classes=%s patch_size=%s",
            self.gt_dir,
            self.image_dir,
            len(self.samples),
            self.classes,
            self.patch_size,
        )

    def _match_image(self, stem: str) -> Path:
        exact = self.image_dir / f"{stem}_te_0.bmp"
        if exact.exists():
            return exact
        candidates = sorted(self.image_dir.glob(f"{stem}_te_0.*"))
        if not candidates:
            candidates = sorted(self.image_dir.glob(f"{stem}*"))
        candidates = [item for item in candidates if item.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}]
        if not candidates:
            raise SegPyError(f"No source image matched .gt3 file stem={stem} in image_dir={self.image_dir}")
        return candidates[0]

    def _load_mask(self, sample: SegSample, image: Image.Image) -> np.ndarray:
        annotation = parse_gt3(sample.label_path, image.size)
        return annotation_to_mask(annotation, image.size)


class YOLOSegDataloader(_PatchSegDataset):
    dataset_type = "yolo"

    def __init__(
        self,
        dataset_yaml: str | Path | None = None,
        image_dir: str | Path | None = None,
        label_dir: str | Path | None = None,
        split: str = "train",
        receptive_field: int = 64,
        patch_size: int | None = None,
        samples_per_image: int = 64,
        positive_ratio: float = 0.5,
        validate_masks: bool = True,
        cache_samples: bool = True,
    ):
        super().__init__(
            receptive_field=receptive_field,
            patch_size=patch_size,
            samples_per_image=samples_per_image,
            positive_ratio=positive_ratio,
            allow_empty_mask=True,
            cache_samples=cache_samples,
        )
        names: list[str] | None = None
        if dataset_yaml is not None:
            if image_dir is not None or label_dir is not None:
                raise SegPyError("Use either --dataset-yaml or --yolo-image-dir/--yolo-label-dir, not both.")
            self.image_dir, self.label_dir, names = load_yolo_dataset_yaml(dataset_yaml, split)
        else:
            if image_dir is None or label_dir is None:
                raise SegPyError("YOLO dataset requires --dataset-yaml or both --yolo-image-dir and --yolo-label-dir.")
            self.image_dir = Path(image_dir)
            self.label_dir = Path(label_dir)
            if not self.label_dir.exists() or not self.label_dir.is_dir():
                raise SegPyError(f"YOLO label directory does not exist or is not a directory: {self.label_dir}")

        image_paths = find_yolo_images(self.image_dir)
        for image_path in image_paths:
            label_path = self.label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise SegPyError(f"YOLO label file does not exist for image {image_path}: {label_path}")
            self.samples.append(SegSample(label_path=label_path, image_path=image_path, name=image_path.stem))

        max_label = 0
        if validate_masks:
            for sample in self.samples:
                _, mask = self.load_full_sample(sample)
                max_label = max(max_label, int(mask.max()))
        else:
            max_label = self._scan_max_yolo_label()

        if names is not None:
            self.classes = ["BG"] + names
            if len(self.classes) <= max_label:
                raise SegPyError(
                    f"YOLO names do not contain enough classes: names={names}, maxMaskLabel={max_label}"
                )
        else:
            self.classes = _classes_from_max_label(max_label)
        self.max_label = max_label
        self._label_range_ready = True

        LOGGER.info(
            "YOLOSegDataloader initialized: image_dir=%s label_dir=%s samples=%s classes=%s patch_size=%s",
            self.image_dir,
            self.label_dir,
            len(self.samples),
            self.classes,
            self.patch_size,
        )

    def _scan_max_yolo_label(self) -> int:
        max_label = 0
        for sample in self.samples:
            text = sample.label_path.read_text(encoding="utf-8-sig")
            for line_number, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id = int(line.split()[0])
                except Exception as exc:
                    raise SegPyError(f"Invalid YOLO class id at {sample.label_path}:{line_number}") from exc
                max_label = max(max_label, class_id + 1)
        return max_label

    def _load_mask(self, sample: SegSample, image: Image.Image) -> np.ndarray:
        annotation = parse_yolo_segmentation_label(
            sample.label_path,
            image.size,
            num_foreground_classes=(len(self.classes) - 1) if self.classes != ["BG", "defect"] else None,
            allow_empty=True,
        )
        return yolo_annotation_to_mask(annotation, image.size)
