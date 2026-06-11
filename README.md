# SegPyProject

PyTorch segmentation project for TeAiFlow `.gt3` annotations and the reverse-engineered `E_TE_V3_SVGF16` network family.

## Files

- `dataloader.py`: `TEDataloader`, loads `.gt3` annotations and real source images.
- `gt3_parser.py`: parses `.gt3` float32 contour points and renders binary masks.
- `seg_models.py`: dynamic `SVGF16` model for receptive fields 32/64/128/256.
- `te_pretrain.py`: loads `E:\TruthEye\TeAiFlow\Application\models\ModelSVGF16`.
- `train.py`: training entrypoint, saves `.pt` checkpoints.
- `export_onnx.py`: exports a trained checkpoint to ONNX.
- `predict.py`: predicts binary masks and rendered overlays.

## Train

```powershell
py E:\TruthEye\WorkDir\SegPyProject\train.py `
  --gt-dir E:\TruthEye\WorkDir\testSegemnt\TestSeg\2 `
  --receptive-field 64
```

## Export ONNX

```powershell
py E:\TruthEye\WorkDir\SegPyProject\export_onnx.py `
  --checkpoint E:\TruthEye\WorkDir\SegPyProject\outputs\checkpoints\svgf16_rf64_best.pt `
  --output E:\TruthEye\WorkDir\SegPyProject\outputs\onnx\svgf16_rf64.onnx
```

```powershell
E:\TruthEye\WorkDir\PythonEnv11-GPU\python.exe export_onnx.py --checkpoint E:\TruthEye\WorkDir\SegPyProject\outputs\checkpoints\svgf16_rf64_best.pt --output E:\TruthEye\WorkDir\SegPyProject\outputs\onnx\svgf16_rf64.onnx
```

## Predict

```powershell
py E:\TruthEye\WorkDir\SegPyProject\predict.py `
  --checkpoint E:\TruthEye\WorkDir\SegPyProject\outputs\checkpoints\svgf16_rf64_best.pt `
  --image E:\TruthEye\WorkDir\testSegemnt\TestSeg\1\SrcImage\1_3@217_te_0.bmp `
  --output-dir E:\TruthEye\WorkDir\SegPyProject\outputs\predict
```

Prediction writes:

- `*_mask.png`: binary foreground mask, `0=BG`, `255=defect`.
- `*_render.png`: red overlay on the original image.

## Notes

- `*_te_0.bmp` files inside TeAiFlow result folders are 80x80 UI thumbnails and are not used for training.
- Training uses real images from `1\SrcImage`.
- The first version is binary segmentation: `BG` vs `defect foreground`.
- Missing dependencies or invalid annotations fail with explicit `errorMessage`; no mock masks or fake successful outputs are produced.
