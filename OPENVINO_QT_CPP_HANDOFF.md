# SVGF16 RF64 ONNX OpenVINO Qt/C++ 推理交接文档

## 1. 模型文件与导出信息

ONNX 模型路径：

```text
E:\TruthEye\WorkDir\SegPyProject\outputs\onnx\svgf16_rf64.onnx
```

当前文件信息：

| 字段 | 值 |
|---|---:|
| 文件大小 | 3,721,041 bytes |
| 修改时间 | 2026-06-11 22:04:14 CST |
| 导出脚本 | `E:\TruthEye\WorkDir\SegPyProject\export_onnx.py` |
| 默认 opset | 17 |
| 模型类型 | `SVGF16` |
| 感受野 | 64 |
| 输出类别数 | 随 checkpoint 变化；当前多类别 `.gt3` 训练为 3 |

导出脚本使用了动态轴：

```python
dynamic_axes={
    "image": {0: "batch", 2: "height", 3: "width"},
    "logits": {0: "batch", 2: "height", 3: "width"},
}
```

因此 ONNX 声明上支持动态 `N/H/W`。但 RF64 网络内部有 3 次 pooling 和 4 次 deconv，C++ 侧仍必须把输入图像 padding 到 stride 16 的整数倍，否则不同后端可能出现输出尺寸不一致或边界误差。

重要提醒：

- 当前 ONNX 可能是由修复 `.gt3` 解析前的旧 checkpoint 导出；旧模型会大面积误检。
- 正式用于 OpenVINO 前，应使用修复后的训练脚本重新训练 `.pt`，再重新导出 ONNX。
- C++ 侧首先验证推理链路与 Python `predict.py` 一致，不要用旧 ONNX 的误检效果判断 OpenVINO 代码错误。

## 2. 输入 Tensor

| 项 | 值 |
|---|---|
| 输入 tensor 名称 | `image` |
| dtype | `float32` |
| layout | `NCHW` |
| shape | `[N, 3, H, W]` |
| RF64 推荐 batch | `N=1` |
| H/W 要求 | padding 后为 16 的整数倍 |

示例：

```text
原图: 1020 x 927
padding 后: W=1024, H=928
OpenVINO 输入 shape: [1, 3, 928, 1024]
```

不要使用 NHWC 输入。若 C++ 使用 `cv::Mat`，需要从 OpenCV 默认的 `HWC/BGR/uint8` 转成 `NCHW/RGB/float32`。

## 3. 图像预处理

Python 参考实现：

- `E:\TruthEye\WorkDir\SegPyProject\predict.py`
  - `predict`
- `E:\TruthEye\WorkDir\SegPyProject\instance_postprocess.py`
  - `postprocess_logits`
  - `render_instances`
- `E:\TruthEye\WorkDir\SegPyProject\utils.py`
  - `read_rgb_image`
  - `image_to_float_array`
  - `pad_array_to_stride`
  - `receptive_field_to_stride`

预处理规则：

| 步骤 | 规则 |
|---|---|
| 读图 | 读取原图，不使用 `*_te_0.bmp` 预览图 |
| 色彩 | RGB |
| OpenCV 注意 | `cv::imread` 是 BGR，必须 `cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB)` |
| resize | 不 resize |
| letterbox | 不做 letterbox |
| padding | 只在右侧和底部 padding 到 stride=16 |
| padding 像素 | edge/replicate padding，对应 Python `np.pad(..., mode="edge")` |
| 归一化 | `float32_value = uint8_value / 255.0` |
| mean/std | 不减 mean，不除 std |
| layout | HWC RGB float32 转 NCHW |

RF64 stride 计算：

```cpp
int stride = 16;
int paddedW = ((origW + stride - 1) / stride) * stride;
int paddedH = ((origH + stride - 1) / stride) * stride;
```

padding 映射：

```text
原图区域: [0, origW) x [0, origH)
padding 区域: x >= origW 或 y >= origH
推理后必须裁回原图区域
```

## 4. 输出 Tensor

| 项 | 值 |
|---|---|
| 输出 tensor 名称 | `logits` |
| dtype | `float32` |
| layout | `NCHW` |
| shape | `[N, num_classes, H, W]` |
| 通道 0 | `BG` 背景 logits |
| 通道 1..N-1 | 训练类别 logits，例如 `.gt3` 当前为 `缺陷1/缺陷2` |

ONNX 内部没有 `Softmax` 或 `Sigmoid` 输出，输出是 raw logits。

Python `predict.py` 当前在推理后先裁回原图尺寸，再调用：

```python
postprocess_logits(logits, classes, conf, iou, max_det)
```

C++ 侧如需完全对齐 Python，应先对所有通道做 softmax，再做 `argmax`：

```cpp
label = argmax_c(logits[c][y][x]);
isForeground = label > 0;
scorePixel = softmax(logits)[label][y][x];
```

对当前 3 类 `.gt3` 模型，`label=0` 是 BG，`label=1` 是 `缺陷1`，`label=2` 是 `缺陷2`。

如果需要显式概率，对 `num_classes` 个通道做 softmax：

```cpp
float maxLogit = max(logits[0..num_classes-1][y][x]);
float denom = 0.0f;
for c in classes:
    prob[c] = std::exp(logits[c][y][x] - maxLogit);
    denom += prob[c];
for c in classes:
    prob[c] /= denom;
```

不要对单个前景通道做 sigmoid，除非后续重新导出的是单通道二分类模型。当前模型是多通道 softmax/argmax 语义。

## 5. 后处理规则

推荐 C++ 后处理流程：

1. 获取 `logits`，确认 shape 为 `[1, numClasses, paddedH, paddedW]`。
2. 对每个像素做 `argmax(channel)` 得到多类别 label。
3. 保存/传出 label 图：背景 `0`，类别从 `1` 开始。
4. 生成 `uint8` 二值 mask：`label > 0` 为 `255`，背景为 `0`。
5. 裁掉右侧/底部 padding，保留 `[0, origW) x [0, origH)`。
6. 对每个前景类别分别做 8 邻域连通域。
7. 每个连通域生成一个 instance，计算 `score`、`polygon`、`bbox`、`area`。
8. 按 score 降序做同类 mask IoU NMS，阈值默认 `iou=0.5`。
9. 保留最多 `maxDet=1000` 个实例，按顺序写入 `instances.png`。
10. 渲染图使用彩色半透明 mask、polygon、bbox、类别名和 score。

轮廓提取建议：

```cpp
cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
```

实例 score 定义：

```text
score = mean(softmax[class_id][y][x]) for all pixels in this instance
```

保存结果规范：

| 文件 | 说明 |
|---|---|
| `*_label.png` | 单通道多类别 label，像素值 `0..numClasses-1` |
| `*_mask.png` | 单通道二值前景，`0/255` |
| `*_instances.png` | 单通道 instance id，`0=BG`，`1..K=instance id` |
| `*_render.png` | 彩色渲染图 |
| `*_results.json` | 完整实例结果 |
| `*_results.txt` | 每行 `cls score x1 y1 x2 y2 ... xn yn` |

JSON 单个 instance 字段：

| 字段 | 含义 |
|---|---|
| `id` | `instances.png` 中对应的实例像素值，范围 `1..K` |
| `cls` | YOLO 风格前景类别 id，背景不占位，所以 `class_id - 1` |
| `class_id` | label 图中的类别 id，`0=BG`，前景从 `1` 开始 |
| `class_name` | checkpoint 中保存的类别名 |
| `score` | 两位小数，实例区域内类别 softmax 均值 |
| `polygon` | 原图像素坐标，不归一化 |
| `bbox` | `[x1, y1, x2, y2]`，右下角为开区间 |
| `area` | 实例像素面积 |

第一版为了和 Python 对齐，不建议默认做面积过滤、形态学开闭运算或连通域合并。若业务上需要接近 TeAiFlow UI 的实例结果，再另行增加后处理参数。

空结果处理：

- mask 全 0 是合法推理结果。
- 但 C++ 接口必须返回状态说明，例如 `hasMask=false` 或 `contourCount=0`。
- 不允许伪造轮廓或静默返回成功但无任何说明。

## 6. 坐标转换规则

当前 Python 推理没有 resize、没有 letterbox、没有 ROI crop，因此坐标转换是直接映射：

```text
模型输入 padded 坐标: (x, y)
原图坐标: (x, y)，前提是 x < origW 且 y < origH
padding 区域: 丢弃
```

mask 裁剪：

```cpp
cv::Rect validRoi(0, 0, origW, origH);
cv::Mat maskOriginal = maskPadded(validRoi).clone();
```

轮廓坐标：

- `findContours` 在裁剪后的 `maskOriginal` 上执行。
- 得到的 contour 点已经是原图坐标。
- bounding box、面积、中心点都按原图像素坐标计算。

如果未来 C++ 侧增加 ROI crop，则必须按以下规则映射：

```text
globalX = roiX + localX
globalY = roiY + localY
```

如果未来增加 resize 或 letterbox，则必须记录 scale 和 padding：

```text
origX = (modelX - padLeft) / scaleX
origY = (modelY - padTop) / scaleY
```

当前版本明确不使用这套 resize/letterbox 规则。

## 7. OpenVINO/Qt C++ 推理接口建议

C++ 封装建议暴露一个结果结构：

```cpp
struct SegInstance {
    int id = 0;
    int cls = 0;                   // YOLO 风格前景类别 id
    int classId = 0;               // label 中的类别 id
    std::string className;
    float score = 0.0f;
    std::vector<cv::Point> polygon;
    cv::Rect bbox;
    int area = 0;
};

struct SegResult {
    bool ok = false;
    std::string errorMessage;
    int width = 0;
    int height = 0;
    cv::Mat label;                 // CV_8UC1 or CV_16UC1, 0..numClasses-1
    cv::Mat mask;                  // CV_8UC1, 0/255, original size
    cv::Mat instanceMask;          // CV_16UC1, 0=BG, 1..K=instance id
    cv::Mat render;                // optional CV_8UC3, BGR or RGB 需在接口中固定
    std::vector<SegInstance> instances;
    float foregroundRatio = 0.0f;
};
```

必须暴露的错误信息：

| 场景 | 必须返回的 `errorMessage` 示例 |
|---|---|
| ONNX 文件不存在 | `ONNX model file does not exist: ...` |
| OpenVINO Core 初始化失败 | `OpenVINO Core initialization failed: ...` |
| 模型读取失败 | `Failed to read ONNX model: ...` |
| 模型编译失败 | `Failed to compile model for device CPU/GPU: ...` |
| 输入 tensor 名称不匹配 | `Input tensor 'image' was not found` |
| 输出 tensor 名称不匹配 | `Output tensor 'logits' was not found` |
| 输入维度不符 | `Invalid input shape, expected NCHW [1,3,H,W], got ...` |
| 图像为空 | `Input image is empty` |
| 图像通道不符 | `Input image must be 3-channel BGR/RGB` |
| padding 后尺寸非法 | `Padded image size must be positive and divisible by 16` |
| 推理失败 | `OpenVINO inference failed: ...` |
| 输出维度不符 | `Invalid output shape, expected [1,numClasses,H,W], got ...` |
| mask 裁剪失败 | `Failed to crop mask back to original image size` |
| 空 mask | `No foreground pixels detected` |
| 无轮廓 | `No valid contours extracted from foreground mask` |
| conf 参数非法 | `conf must be in [0,1], got ...` |
| iou 参数非法 | `iou must be in [0,1], got ...` |
| maxDet 参数非法 | `max_det must be in [1,65535], got ...` |
| instance id 溢出 | `Instance count exceeds uint16 instance mask capacity` |

空 mask 和无轮廓不一定是程序错误，但必须让调用方知道。

## 8. Python 参考位置

| 功能 | Python 文件 | 函数/类 |
|---|---|---|
| 模型定义 | `E:\TruthEye\WorkDir\SegPyProject\seg_models.py` | `SVGF16` |
| checkpoint 加载 | `E:\TruthEye\WorkDir\SegPyProject\seg_models.py` | `build_svgf16_from_checkpoint` |
| ONNX 导出 | `E:\TruthEye\WorkDir\SegPyProject\export_onnx.py` | `export_onnx` |
| 图像读取 | `E:\TruthEye\WorkDir\SegPyProject\utils.py` | `read_rgb_image` |
| 图像归一化 | `E:\TruthEye\WorkDir\SegPyProject\utils.py` | `image_to_float_array` |
| stride padding | `E:\TruthEye\WorkDir\SegPyProject\utils.py` | `pad_array_to_stride` |
| 预测主流程 | `E:\TruthEye\WorkDir\SegPyProject\predict.py` | `predict` |
| 实例后处理 | `E:\TruthEye\WorkDir\SegPyProject\instance_postprocess.py` | `postprocess_logits` |
| 渲染实例图 | `E:\TruthEye\WorkDir\SegPyProject\instance_postprocess.py` | `render_instances` |
| TXT 保存 | `E:\TruthEye\WorkDir\SegPyProject\instance_postprocess.py` | `write_instances_txt` |
| 标注解析验证 | `E:\TruthEye\WorkDir\SegPyProject\gt3_parser.py` | `parse_gt3`, `annotation_to_mask` |

## 9. 测试图与期望结果

测试图 1：

```text
E:\TruthEye\WorkDir\testSegemnt\TestSeg\1\SrcImage\1_3@217_te_0.bmp
```

图像尺寸：

```text
1020 x 927
```

padding 后输入：

```text
[1, 3, 928, 1024]
```

基于修复后的 `TestSeg\2\1_3@217.gt3`，期望标注分布：

| 区域 | 期望 bbox 约值 `(x1,y1,x2,y2)` | 说明 |
|---:|---|---|
| 1 | `(166,122,219,221)` | R87 附近小矩形 |
| 2 | `(167,388,219,489)` | R88 附近小矩形 |
| 3 | `(164,652,218,753)` | R89 附近小矩形 |
| 4 | `(908,457,1018,648)` | 右侧 R79 附近元件 |
| 5 | `(506,449,740,691)` | R82 附近较大元件 |

期望现象：

- mask 前景应集中在上述几个元件区域。
- 前景面积比例应是小比例，标注 mask 约 `5.67%`。
- 如果整张图出现大片红色，例如超过 `20%`，优先检查是否使用了旧 checkpoint/旧 ONNX 或预处理通道顺序错误。

测试图 2：

```text
E:\TruthEye\WorkDir\testSegemnt\TestSeg\1\SrcImage\1_1_20250120233915_1@146_te_0.bmp
```

图像尺寸：

```text
793 x 598
```

padding 后输入：

```text
[1, 3, 608, 800]
```

基于修复后的 `TestSeg\2\1_1_20250120233915_1@146.gt3`，期望标注分布：

| 区域 | 期望 bbox 约值 `(x1,y1,x2,y2)` |
|---:|---|
| 1 | `(1,484,35,555)` |
| 2 | `(751,2,791,70)` |

期望现象：

- 只应在左右/边缘局部位置出现前景。
- 标注 mask 前景比例约 `0.91%`。

## 10. C++ 与 Python 对齐检查清单

实现完成后，用同一张图片分别跑 Python 和 OpenVINO C++，检查：

| 检查项 | 期望 |
|---|---|
| 输入尺寸 | C++ padding 后尺寸与 Python 一致 |
| 输入数值 | RGB、`/255.0`、NCHW |
| 输出 shape | `[1,numClasses,paddedH,paddedW]` |
| mask 裁剪 | 输出 mask 与原图尺寸一致 |
| mask 二值 | 只有 `0/255` |
| instance mask | `0=BG`，`1..K` 与 JSON 中的 `id` 一一对应 |
| instance 字段 | 每个实例都有 `cls/class_name/score/polygon/bbox/area` |
| 前景比例 | 与 Python 输出接近 |
| 渲染位置 | 红色区域在同一批元件附近 |

若 C++ 与 Python 差异很大，排查顺序：

1. 是否使用修复后重新训练、重新导出的 ONNX。
2. 是否把 OpenCV BGR 转成 RGB。
3. 是否只做 `/255.0`，没有额外 mean/std。
4. 是否 NCHW，而不是 NHWC。
5. 是否 padding 到 stride 16，并在输出后裁回原图尺寸。
6. 是否对多通道 logits 做 argmax/softmax，而不是单通道 sigmoid。
