from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from utils import DependencyError, SegPyError, receptive_field_to_stride

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - depends on runtime env
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from te_pretrain import DEFAULT_PRETRAIN_PATH, load_svgf16_pretrained_weights


RF_CHANNELS = {
    32: [16, 32, 64],
    64: [16, 32, 64, 128],
    128: [16, 32, 64, 128, 256],
    256: [16, 32, 64, 128, 256, 512],
}


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise DependencyError(
            "PyTorch is required but is not installed. Install torch before using SVGF16."
        )


class ConvBnAct(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        _require_torch()
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Any) -> Any:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DeconvBnAct(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, in_channels: int, out_channels: int):
        _require_torch()
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Any) -> Any:
        return F.relu(self.bn(self.deconv(x)), inplace=True)


class SVGF16(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(
        self,
        receptive_field: int = 64,
        in_channels: int = 3,
        num_classes: int = 2,
        pretrained_path: str | Path = DEFAULT_PRETRAIN_PATH,
        load_pretrained: bool = True,
    ):
        _require_torch()
        super().__init__()
        if receptive_field not in RF_CHANNELS:
            raise SegPyError(
                f"Unsupported receptive_field={receptive_field}. Supported values: {sorted(RF_CHANNELS)}"
            )
        if in_channels != 3:
            raise SegPyError("SVGF16 currently supports in_channels=3 because ModelSVGF16 pretrain is RGB.")
        if num_classes < 2:
            raise SegPyError(f"num_classes must be >= 2, got {num_classes}")

        self.receptive_field = int(receptive_field)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.channels = RF_CHANNELS[self.receptive_field]
        self.output_stride = receptive_field_to_stride(self.receptive_field)

        self.input_bn = nn.BatchNorm2d(in_channels)
        self.encoder = nn.ModuleList()
        self.encoder_names: list[list[str]] = []
        self._named_svgf16: OrderedDict[str, Any] = OrderedDict()
        self._named_svgf16["bn0"] = self.input_bn

        cov_index = 1
        current_in = in_channels
        for stage_index, out_channels in enumerate(self.channels):
            stage = nn.ModuleList()
            names: list[str] = []
            for conv_in_stage in range(4 if stage_index == 0 else 3):
                if stage_index == 0 and conv_in_stage == 0:
                    kernel = 7
                    stride = 2
                    conv_in = current_in
                else:
                    kernel = 3
                    stride = 1
                    conv_in = current_in
                layer = ConvBnAct(conv_in, out_channels, kernel_size=kernel, stride=stride)
                name = f"cov{cov_index}"
                self._named_svgf16[name] = layer
                stage.append(layer)
                names.append(name)
                current_in = out_channels
                cov_index += 1
            self.encoder.append(stage)
            self.encoder_names.append(names)

        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(len(self.channels) - 1)])

        self.skip_up_blocks = nn.ModuleList()
        self.skip_dec_blocks = nn.ModuleList()
        current_channels = self.channels[-1]
        for skip_channels in reversed(self.channels[:-1]):
            self.skip_up_blocks.append(DeconvBnAct(current_channels, skip_channels))
            self.skip_dec_blocks.append(ConvBnAct(skip_channels * 2, skip_channels * 2, kernel_size=3))
            current_channels = skip_channels * 2

        self.final_up = DeconvBnAct(current_channels, 32)
        self.final_dec = ConvBnAct(32, 32, kernel_size=3)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=3, padding=1, bias=False)

        if load_pretrained:
            self.pretrain_report = load_svgf16_pretrained_weights(self, pretrained_path)
        else:
            self.pretrain_report = {"loaded": 0, "skipped": 0, "available": 0}

    def named_svgf16_modules(self) -> OrderedDict[str, Any]:
        return self._named_svgf16

    @staticmethod
    def _match_spatial(x: Any, reference: Any) -> Any:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: Any) -> Any:
        input_size = x.shape[-2:]
        x = self.input_bn(x)
        skips: list[Any] = []
        for stage_index, stage in enumerate(self.encoder):
            for layer in stage:
                x = layer(x)
            skips.append(x)
            if stage_index < len(self.pools):
                x = self.pools[stage_index](x)

        for up, dec, skip in zip(self.skip_up_blocks, self.skip_dec_blocks, reversed(skips[:-1])):
            x = up(x)
            x = self._match_spatial(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.final_up(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.final_dec(x)
        logits = self.classifier(x)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def build_svgf16_from_checkpoint(checkpoint: dict[str, Any], device: str = "cpu") -> SVGF16:
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise SegPyError("Checkpoint is missing model_config")
    model = SVGF16(
        receptive_field=int(config["receptive_field"]),
        in_channels=int(config.get("in_channels", 3)),
        num_classes=int(config.get("num_classes", 2)),
        load_pretrained=False,
    )
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise SegPyError("Checkpoint is missing model_state_dict")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

