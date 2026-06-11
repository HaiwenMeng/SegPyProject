from __future__ import annotations

import argparse
import logging
from pathlib import Path

from seg_models import build_svgf16_from_checkpoint
from utils import SegPyError, ensure_dir, require_torch, setup_logging, write_json_result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained SVGF16 .pt checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def export_onnx(args: argparse.Namespace) -> dict[str, object]:
    torch = require_torch()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SegPyError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model = build_svgf16_from_checkpoint(checkpoint, device=args.device)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    if args.input_size <= 0:
        raise SegPyError(f"input-size must be > 0, got {args.input_size}")

    dummy = torch.randn(1, 3, args.input_size, args.input_size, device=args.device)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )

    if args.verify:
        try:
            import onnx  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise SegPyError("ONNX verification requested but the onnx package is not installed.") from exc
        model_proto = onnx.load(str(output_path))
        onnx.checker.check_model(model_proto)

    return {"ok": True, "onnx": str(output_path), "checkpoint": str(checkpoint_path)}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        write_json_result(export_onnx(args))
        return 0
    except Exception as exc:
        logging.getLogger("segpy.export_onnx").exception("ONNX export failed")
        write_json_result({"ok": False, "errorMessage": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

