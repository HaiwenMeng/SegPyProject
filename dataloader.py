from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from gt3_parser import annotation_to_mask, parse_gt3
from utils import DependencyError, SegPyError, image_to_float_array, read_rgb_image, receptive_field_to_patch_size

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - depends on runtime env
    torch = None  # type: ignore
    Dataset = object  # type: ignore


LOGGER = logging.getLogger("segpy.dataloader")


@dataclass(frozen=True)
class TESample:
    gt_path: Path
    image_path: Path
    name: str


def _require_torch() -> None:
    if torch is None:
        raise DependencyError(
            "PyTorch is required but is not installed. Install torch before using TEDataloader."
        )


class TEDataloader(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        gt_dir: str | Path,
        image_dir: str | Path | None = None,
        receptive_field: int = 64,
        patch_size: int | None = None,
        samples_per_image: int = 64,
        positive_ratio: float = 0.5,
        validate_masks: bool = True,
    ):
        _require_torch()
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

        self.receptive_field = int(receptive_field)
        self.patch_size = int(patch_size or receptive_field_to_patch_size(self.receptive_field))
        self.samples_per_image = int(samples_per_image)
        self.positive_ratio = float(positive_ratio)
        if self.samples_per_image <= 0:
            raise SegPyError(f"samples_per_image must be > 0, got {samples_per_image}")
        if not 0.0 <= self.positive_ratio <= 1.0:
            raise SegPyError(f"positive_ratio must be in [0,1], got {positive_ratio}")
        if self.patch_size <= 0:
            raise SegPyError(f"patch_size must be > 0, got {self.patch_size}")

        gt_paths = sorted(self.gt_dir.glob("*.gt3"))
        if not gt_paths:
            raise SegPyError(f"No .gt3 files found in gt_dir: {self.gt_dir}")

        self.samples: list[TESample] = []
        for gt_path in gt_paths:
            image_path = self._match_image(gt_path.stem)
            self.samples.append(TESample(gt_path=gt_path, image_path=image_path, name=gt_path.stem))

        self._sample_cache: dict[Path, tuple[Image.Image, np.ndarray]] = {}
        if validate_masks:
            for sample in self.samples:
                _, mask = self.load_full_sample(sample)
                if int(mask.sum()) <= 0:
                    raise SegPyError(f"Mask is empty for sample: {sample.gt_path}")

        LOGGER.info(
            "TEDataloader initialized: gt_dir=%s image_dir=%s samples=%s patch_size=%s",
            self.gt_dir,
            self.image_dir,
            len(self.samples),
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

    def __len__(self) -> int:
        return len(self.samples) * self.samples_per_image

    def load_full_sample(self, sample: TESample) -> tuple[Image.Image, np.ndarray]:
        cached = self._sample_cache.get(sample.gt_path)
        if cached is not None:
            return cached
        image = read_rgb_image(sample.image_path)
        annotation = parse_gt3(sample.gt_path, image.size)
        mask = annotation_to_mask(annotation, image.size)
        self._sample_cache[sample.gt_path] = (image, mask)
        return image, mask

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
        if int(mask.sum()) <= 0:
            raise SegPyError(f"Mask is empty after preprocessing for sample: {sample.gt_path}")
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
            "gt_path": str(sample.gt_path),
            "crop": torch.tensor([x0, y0, x1, y1], dtype=torch.int64),
        }
