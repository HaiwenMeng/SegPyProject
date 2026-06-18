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
- `eval.py`: evaluates `.pt` checkpoints on YOLO segmentation datasets and writes IoU metrics plus mosaic previews.
- `eval_full.py`: evaluates a checkpoint on full annotated images and compares raw logits with postprocessed masks.
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
  --max-det 1000 `
  --min-pixel 5 `
  --mask-thresh 0.0
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
  --max-det 1000 `
  --min-pixel 5 `
  --mask-thresh 0.0
```

Prediction writes:

- `*_label.png`: multiclass label image, `0=BG`, foreground classes start at `1`.
- `*_mask.png`: binary foreground mask, `0=BG`, `255=defect`.
- `*_instances.png`: single-channel instance id image, `0=BG`, `1..K=instance id`.
- `*_render.png`: color overlay with polygon, bbox, class name, and score.
- `*_results.json`: full per-instance result. Each instance includes `id`, `cls`, `class_id`, `class_name`, `score`, `polygon`, `bbox`, and `area`.
- `*_results.txt`: written when `--save-txt true`; each line is `cls score x1 y1 x2 y2 ... xn yn`.

`predict.py` is still running a semantic segmentation model. The YOLO-seg style instances are derived from multiclass `argmax` labels by connected components and contours. `score` is the mean softmax probability inside each instance region, rounded to two decimals.

Use `--min-pixel N` to discard connected foreground components smaller than `N` pixels during instance postprocessing. The typo-compatible alias `--min-pxiel` is also accepted.

Use `--mask-thresh T` to discard foreground pixels whose winning class softmax probability is lower than `T` before connected-component extraction. `--mask-thresh 0.0` keeps the original argmax behavior. For noisy texture images, start with `--mask-thresh 0.9` to `0.97`, then tune `--min-pixel` and `--max-det`.

`--conf` is instance-level filtering: after a connected component is formed, its `score` is the mean softmax probability of that component. `--mask-thresh` is pixel-level filtering and is usually the more effective switch for suppressing dense tiny false positives.

## YOLO Seg Evaluation

Use `eval.py` for regular YOLO segmentation dataset evaluation. It uses tiled batch inference and does not keep full-image float logits in memory.

```powershell
py E:\TruthEye\WorkDir\SegPyProject\eval.py `
  --checkpoint E:\path\to\best.pt `
  --dataset-yaml E:\path\to\dataset.yaml `
  --split val `
  --output-dir E:\TruthEye\WorkDir\SegPyProject\outputs\eval `
  --tile-size 512 `
  --tile-overlap 64 `
  --batch-size 4 `
  --max-vis 24
```

Outputs:

- `summary.json`: global pixel metrics, foreground IoU/precision/recall/dice, mIoU, per-class metrics, and confusion matrix.
- `per_image_metrics.csv`: one row per image.
- `per_class_metrics.csv`: aggregate metrics for each class.
- `mosaic_*.jpg`: preview sheets where each sample block contains `Image`, `GT`, `Pred`, and `Overlay`.

For quick checks use `--max-images 1`. If dense false positives are visible, compare runs with `--mask-thresh 0.0` and `--mask-thresh 0.95`.

## Full-image evaluation

Use `eval_full.py` when patch training metrics look good but full-image prediction is bad. By default it uses tiled inference, does not cache full images, does not save intermediate images, and evaluates only raw model logits:

- `raw`: direct `argmax(logits)` from the model, before instance postprocessing.
- `post`: optional final label map after `--mask-thresh`, `--min-pixel`, `--conf`, `--iou`, and `--max-det`; enable it with `--eval-post true`.

```powershell
py E:\TruthEye\WorkDir\SegPyProject\eval_full.py `
  --checkpoint E:\TruthEye\WorkDir\SegPyProject\outputs\checkpoints\svgf16_rf64_best.pt `
  --dataset-type gt3 `
  --gt-dir E:\TruthEye\WorkDir\testSegemnt\TestSeg\2 `
  --output-dir E:\TruthEye\WorkDir\SegPyProject\outputs\eval_full `
  --max-images 20 `
  --mask-thresh 0.0 `
  --min-pixel 1
```

YOLO segmentation datasets use the same dataset arguments as `train.py`:

```powershell
py E:\TruthEye\WorkDir\SegPyProject\eval_full.py `
  --checkpoint E:\path\to\best.pt `
  --dataset-type yolo `
  --dataset-yaml E:\path\to\dataset.yaml `
  --split train `
  --output-dir E:\TruthEye\WorkDir\SegPyProject\outputs\eval_full `
  --max-images 20
```

Outputs:

- `summary.json`: aggregate raw/post metrics and per-image details.
- `per_image_metrics.csv`: per-image raw/post metrics for spreadsheet comparison.
- `images/*_gt_label.png`: ground-truth label map, only when `--save-images true`.
- `images/*_raw_label.png`: direct model argmax label map, only when `--save-images true`.
- `images/*_post_label.png`: postprocessed label map, only when both `--save-images true` and `--eval-post true`.
- `images/*_raw_error.png` and `*_post_error.png`: error maps where green=true foreground, red=false positive, blue=false negative, yellow=foreground class mismatch.

Recommended diagnostic order for large images:

1. Run `--max-images 20` with default `--eval-post false` to check raw model quality first.
2. If raw metrics are good, run the same subset with `--eval-post true --mask-thresh ... --min-pixel ...` to debug postprocessing.
3. If memory is still high, set `--tile-size 512 --tile-overlap 64` explicitly and keep `--save-images false`.

## Notes

- `*_te_0.bmp` files inside TeAiFlow result folders are 80x80 UI thumbnails and are not used for training.
- Training uses real images from `1\SrcImage`.
- Training is multiclass for both `.gt3` and YOLO segmentation. The binary mask is only a prediction-time derived output where `label > 0`.
- Missing dependencies or invalid annotations fail with explicit `errorMessage`; no mock masks or fake successful outputs are produced.
