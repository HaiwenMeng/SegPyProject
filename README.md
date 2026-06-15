# SegPyProject

PyTorch segmentation project for TeAiFlow `.gt3` annotations, YOLO segmentation datasets, and the reverse-engineered `E_TE_V3_SVGF16` network family.

## Files

- `dataloader.py`: `TEDataloader` and `YOLOSegDataloader`, loads `.gt3` or YOLO segmentation datasets.
- `gt3_parser.py`: parses `.gt3` float32 contour points and renders multiclass masks.
- `yolo_parser.py`: parses YOLO segmentation polygon labels.
- `seg_models.py`: dynamic `SVGF16` model for receptive fields 32/64/128/256.
- `te_pretrain.py`: loads `E:\TruthEye\TeAiFlow\Application\models\ModelSVGF16`.
- `train.py`: training entrypoint, saves `.pt` checkpoints.
- `export_onnx.py`: exports a trained checkpoint to ONNX.
- `instance_postprocess.py`: converts semantic logits into lightweight YOLO-seg style instances.
- `predict.py`: predicts label maps, binary masks, instance masks, JSON/TXT results, and rendered overlays.

## Train TeAiFlow .gt3

```powershell
py E:\TruthEye\WorkDir\SegPyProject\train.py `
  --gt-dir E:\TruthEye\WorkDir\testSegemnt\TestSeg\2 `
  --receptive-field 64
```

The `.gt3` flow is multiclass. For the current TestSeg project the classes are:

```text
0=BG, 1=缺陷1, 2=缺陷2
```

## Train YOLO segmentation

Using `dataset.yaml`:

```powershell
py E:\TruthEye\WorkDir\SegPyProject\train.py `
  --dataset-type yolo `
  --dataset-yaml E:\path\to\dataset.yaml `
  --split train `
  --receptive-field 64
```

Using explicit directories:

```powershell
py E:\TruthEye\WorkDir\SegPyProject\train.py `
  --dataset-type yolo `
  --yolo-image-dir E:\path\to\images\train `
  --yolo-label-dir E:\path\to\labels\train `
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
  --output-dir E:\TruthEye\WorkDir\SegPyProject\outputs\predict_instances `
  --conf 0.2 `
  --iou 0.5 `
  --save-txt true `
  --show false `
  --max-det 1000
```

Folder prediction uses the same output format:

```powershell
py E:\TruthEye\WorkDir\SegPyProject\predict.py `
  --checkpoint E:\TruthEye\WorkDir\SegPyProject\outputs\checkpoints\svgf16_rf64_best.pt `
  --folder E:\TruthEye\WorkDir\testSegemnt\TestSeg\1\SrcImage `
  --output-dir E:\TruthEye\WorkDir\SegPyProject\outputs\predict_instances `
  --conf 0.2 `
  --iou 0.5 `
  --save-txt true `
  --show false `
  --max-det 1000
```

Prediction writes:

- `*_label.png`: multiclass label image, `0=BG`, foreground classes start at `1`.
- `*_mask.png`: binary foreground mask, `0=BG`, `255=defect`.
- `*_instances.png`: single-channel instance id image, `0=BG`, `1..K=instance id`.
- `*_render.png`: color overlay with polygon, bbox, class name, and score.
- `*_results.json`: full per-instance result. Each instance includes `id`, `cls`, `class_id`, `class_name`, `score`, `polygon`, `bbox`, and `area`.
- `*_results.txt`: written when `--save-txt true`; each line is `cls score x1 y1 x2 y2 ... xn yn`.

`predict.py` is still running a semantic segmentation model. The YOLO-seg style instances are derived from multiclass `argmax` labels by connected components and contours. `score` is the mean softmax probability inside each instance region, rounded to two decimals.

## Notes

- `*_te_0.bmp` files inside TeAiFlow result folders are 80x80 UI thumbnails and are not used for training.
- Training uses real images from `1\SrcImage`.
- Training is multiclass for both `.gt3` and YOLO segmentation. The binary mask is only a prediction-time derived output where `label > 0`.
- Missing dependencies or invalid annotations fail with explicit `errorMessage`; no mock masks or fake successful outputs are produced.
