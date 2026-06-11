from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils import SegPyError


DEFAULT_PRETRAIN_PATH = Path(r"E:\TruthEye\TeAiFlow\Application\models\ModelSVGF16")
LOGGER = logging.getLogger("segpy.pretrain")


@dataclass(frozen=True)
class PretrainTensor:
    name: str
    shape: tuple[int, int, int, int]
    array: np.ndarray


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _fixed_ascii(data: bytes, offset: int, size: int) -> str:
    chunk = data[offset : offset + size]
    end = chunk.find(b"\x00")
    if end >= 0:
        chunk = chunk[:end]
    return chunk.decode("ascii", errors="replace").strip()


def _prod(shape: tuple[int, int, int, int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def load_modelsvgf16(path: str | Path = DEFAULT_PRETRAIN_PATH) -> dict[str, PretrainTensor]:
    model_path = Path(path)
    if not model_path.exists():
        raise SegPyError(f"Pretrained ModelSVGF16 file does not exist: {model_path}")
    if not model_path.is_file():
        raise SegPyError(f"Pretrained ModelSVGF16 path is not a file: {model_path}")

    data = model_path.read_bytes()
    if len(data) < 4:
        raise SegPyError(f"Pretrained ModelSVGF16 file is too small: {model_path}")

    count = _u32(data, 0)
    record_size = 84
    weight_start = 4 + count * record_size
    if count <= 0 or weight_start >= len(data):
        raise SegPyError(
            f"Invalid ModelSVGF16 header: count={count}, weightStart={weight_start}, size={len(data)}"
        )

    cursor = weight_start
    tensors: dict[str, PretrainTensor] = {}
    for index in range(count):
        offset = 4 + index * record_size
        shape = tuple(_u32(data, offset + i * 4) for i in range(4))  # type: ignore[assignment]
        name = _fixed_ascii(data, offset + 20, 64)
        if not name:
            raise SegPyError(f"ModelSVGF16 tensor record has empty name at index={index}")
        elements = _prod(shape)
        byte_count = elements * 4
        if cursor + byte_count > len(data):
            raise SegPyError(
                f"ModelSVGF16 tensor exceeds file: name={name}, index={index}, offset={cursor}"
            )
        array = np.frombuffer(data, dtype="<f4", count=elements, offset=cursor).copy().reshape(shape)
        tensors[name] = PretrainTensor(name=name, shape=shape, array=array)
        cursor += byte_count

    if cursor != len(data):
        raise SegPyError(
            f"ModelSVGF16 tensor bytes do not consume file exactly: cursor={cursor}, size={len(data)}"
        )
    return tensors


def _copy_tensor(torch: Any, destination: Any, source: np.ndarray, name: str) -> bool:
    tensor = torch.from_numpy(source)
    if tuple(destination.shape) != tuple(tensor.shape):
        LOGGER.info(
            "Skip pretrained tensor %s because shape differs: model=%s pretrain=%s",
            name,
            tuple(destination.shape),
            tuple(tensor.shape),
        )
        return False
    with torch.no_grad():
        destination.copy_(tensor.to(device=destination.device, dtype=destination.dtype))
    return True


def load_svgf16_pretrained_weights(model: Any, path: str | Path = DEFAULT_PRETRAIN_PATH) -> dict[str, int]:
    torch = __import__("torch")
    tensors = load_modelsvgf16(path)
    loaded = 0
    skipped = 0

    def get(name: str) -> np.ndarray | None:
        item = tensors.get(name)
        return None if item is None else item.array

    for name, module in model.named_svgf16_modules().items():
        if name == "bn0":
            prefix = "bn0"
            weight = get(f"{prefix}.weight")
            bias = get(f"{prefix}.bias")
            mean = get(f"{prefix}.running_mean")
            var = get(f"{prefix}.running_var")
            if weight is None or bias is None or mean is None or var is None:
                skipped += 4
                continue
            loaded += int(_copy_tensor(torch, module.weight, weight.reshape(-1), f"{prefix}.weight"))
            loaded += int(_copy_tensor(torch, module.bias, bias.reshape(-1), f"{prefix}.bias"))
            loaded += int(_copy_tensor(torch, module.running_mean, mean.reshape(-1), f"{prefix}.running_mean"))
            loaded += int(_copy_tensor(torch, module.running_var, var.reshape(-1), f"{prefix}.running_var"))
            continue

        if not name.startswith("cov"):
            continue
        conv_name = f"{name}.weight"
        bn_name = f"bn{name[3:]}"
        conv_weight = get(conv_name)
        bn_weight = get(f"{bn_name}.weight")
        bn_bias = get(f"{bn_name}.bias")
        bn_mean = get(f"{bn_name}.running_mean")
        bn_var = get(f"{bn_name}.running_var")
        if conv_weight is None or bn_weight is None or bn_bias is None or bn_mean is None or bn_var is None:
            skipped += 5
            LOGGER.info("No pretrained tensors found for %s/%s; keeping initialized weights", name, bn_name)
            continue

        out_channels = module.conv.out_channels
        in_channels = module.conv.in_channels
        kernel_h, kernel_w = module.conv.kernel_size
        standard_shape = (out_channels, in_channels, kernel_h, kernel_w)
        flat_shape = (1, out_channels * in_channels, kernel_h, kernel_w)
        if conv_weight.shape == standard_shape:
            conv_source = conv_weight
        elif conv_weight.shape == flat_shape:
            conv_source = conv_weight.reshape(standard_shape)
        else:
            conv_source = None
            skipped += 1
            LOGGER.info(
                "Skip %s due to pretrain shape mismatch: expected %s or %s got=%s",
                conv_name,
                standard_shape,
                flat_shape,
                conv_weight.shape,
            )
        if conv_source is not None:
            loaded += int(_copy_tensor(torch, module.conv.weight, conv_source, conv_name))
        loaded += int(_copy_tensor(torch, module.bn.weight, bn_weight.reshape(-1), f"{bn_name}.weight"))
        loaded += int(_copy_tensor(torch, module.bn.bias, bn_bias.reshape(-1), f"{bn_name}.bias"))
        loaded += int(_copy_tensor(torch, module.bn.running_mean, bn_mean.reshape(-1), f"{bn_name}.running_mean"))
        loaded += int(_copy_tensor(torch, module.bn.running_var, bn_var.reshape(-1), f"{bn_name}.running_var"))

    LOGGER.info("Pretrained ModelSVGF16 load result: loaded=%s skipped=%s path=%s", loaded, skipped, path)
    return {"loaded": loaded, "skipped": skipped, "available": len(tensors)}
