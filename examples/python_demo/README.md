# VisionEngine Python 测试示例

## 概述

本目录提供VisionEngine视觉推理引擎的Python测试接口示例。由于VisionEngine核心库是C++实现的，本示例通过以下两种方式测试：

1. **ONNX Runtime直接调用**：使用VisionEngine后端支持的ONNX模型进行推理测试
2. **模拟模式**：当ONNX Runtime不可用时，使用模拟数据进行功能验证

## 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖项

- `numpy` - 数值计算库
- `onnxruntime` - ONNX Runtime推理引擎 (可选)
- `Pillow` - 图像处理库

```bash
# 完整安装
pip install numpy onnxruntime pillow

# GPU版本 (需要CUDA支持)
pip install numpy onnxruntime-gpu pillow
```

## 使用方法

### 1. 基本测试

```bash
cd python_demo
python test_vision_engine.py
```

### 2. 指定模型测试

```bash
python test_vision_engine.py --model ./models/yolov8n.onnx
```

### 3. 运行特定测试

```python
from test_vision_engine import VisionEngineSimulator, test_single_image

# 创建引擎
engine = VisionEngineSimulator("path/to/model.onnx")

# 测试单张图像
detections = test_single_image(engine, "path/to/image.jpg")
```

## 支持的模型

| 模型名称 | 类型 | 说明 |
|---------|------|------|
| yolov8n.onnx | 目标检测 | YOLOv8nano模型 |
| yolov5s.onnx | 目标检测 | YOLOv5s模型 |
| ssd_mobilenet_v1.onnx | 目标检测 | SSDMobileNet模型 |
| resnet50.onnx | 图像分类 | ResNet50模型 |

模型文件位于：`examples/qt6_demo/models/`

## 功能特性

### 图像预处理
- 自动加载RGB图像
- 调整至模型输入尺寸 (640x640)
- 像素值归一化 [0,1]
- HWC -> CHW 转换

### NMS算法模块

支持多种NMS（非极大值抑制）算法：

```python
from test_vision_engine import nms, soft_nms, classwise_nms, ciou_nms, FastNMS

# 标准NMS
boxes = np.array([[x1, y1, x2, y2], ...])
scores = np.array([0.9, 0.8, ...])
keep_indices = nms(boxes, scores, iou_threshold=0.45)

# Soft-NMS (高斯衰减)
keep_indices = soft_nms(boxes, scores, iou_threshold=0.45, sigma=0.5)

# 按类别NMS
keep_indices = classwise_nms(detections, iou_threshold=0.45)

# CIoU-NMS
keep_indices = ciou_nms(detections, iou_threshold=0.45)

# Fast-NMS (空间哈希加速)
fast_nms = FastNMS(iou_threshold=0.45, max_detections=100, grid_size=64)
result = fast_nms.process(detections)
```

### IoU计算模块

支持多种IoU变体：

```python
from test_vision_engine import calculate_iou, calculate_giou, calculate_diou, calculate_ciou

# 标准IoU
iou = calculate_iou(box_a, box_b)

# GIoU (Generalized IoU)
giou = calculate_giou(box_a, box_b)

# DIoU (Distance IoU)
diou = calculate_diou(box_a, box_b)

# CIoU (Complete IoU)
ciou = calculate_ciou(box_a, box_b)
```

### YOLO解析器

YOLOv8/v7/v5输出解析：

```python
from test_vision_engine import YOLOParser

# 创建解析器
parser = YOLOParser()
parser.set_config({
    'confidence_threshold': 0.5,
    'nms_threshold': 0.45,
    'num_classes': 80,
    'use_ciou': True
})

# 解析输出
detections = parser.parse_yolov8(
    output_data,
    output_shape,
    original_width,
    original_height,
    scale_x,
    scale_y,
    pad_x,
    pad_y
)
```

### 多尺度检测器

多尺度特征融合检测：

```python
from test_vision_engine import MultiScaleDetector

detector = MultiScaleDetector()
detector.set_config({
    'confidence_threshold': 0.5,
    'nms_threshold': 0.45,
    'strides': [8, 16, 32]  # P3, P4, P5
})

# 执行多尺度检测
detections = detector.detect(outputs, shapes)
```

### 测试时增强 (TTA)

通过多尺度、翻转等增强提升检测精度：

```python
from test_vision_engine import TTAEngine

tta = TTAEngine()
tta.set_config({
    'horizontal_flip': True,
    'multi_scale': True,
    'scales': [0.83, 1.0, 1.17]
})

# 执行TTA推理
result = tta.process(image, width, height, infer_func)
```

### 置信度过滤

```python
from test_vision_engine import filter_by_confidence

# 过滤低置信度检测
filtered = filter_by_confidence(detections, threshold=0.5)
```

### 检测可视化

```python
from test_vision_engine import DetectionVisualizer

# 绘制检测框
visualized = DetectionVisualizer.draw_detections(image_data, width, height, detections)
```

### 内存池

```python
from test_vision_engine import TensorMemoryPool

# 获取内存池实例
pool = TensorMemoryPool.instance()

# 分配张量内存
tensor = pool.allocate_tensor(640, 640, 3)

# 释放内存
pool.deallocate(tensor)
```

### 测试模式
- 单张图像推理测试
- 批量图像推理测试
- 性能基准测试

## 目录结构

```
python_demo/
├── README.md              # 本文档
├── requirements.txt       # Python依赖
├── test_vision_engine.py  # 主测试脚本
├── models/                # 模型目录 (可选)
│   └── yolov8n.onnx
└── test_images/           # 测试图像目录
```

## 与C++ API对应关系

| Python函数 | C++ API | 说明 |
|-----------|---------|------|
| `VisionEngineSimulator()` | `InferenceEngine()` | 创建推理引擎 |
| `load_model()` | `LoadModel()` | 加载模型 |
| `infer()` | `Infer()` | 执行推理 |
| `SetConfidenceThreshold()` | `SetConfidenceThreshold()` | 设置置信度阈值 |
| `SetNMSThreshold()` | `SetNMSThreshold()` | 设置NMS阈值 |
| `YOLOParser` | `YOLOParser` | YOLO输出解析器 |
| `MultiScaleDetector` | `MultiScaleDetector` | 多尺度检测器 |
| `FastNMS` | `FastNMS` | 快速NMS (O(n)) |
| `TTAEngine` | `TTAEngine` | 测试时增强 |
| `TensorMemoryPool` | `TensorMemoryPool` | 内存池管理 |

## 性能优化预期

| 优化项 | 预期提升 |
|--------|----------|
| CIoU-NMS | mAP +2-3% |
| Fast-NMS | NMS时间 -50% |
| 多尺度融合 | mAP +5-10% |
| TTA增强 | mAP +2-3% |

## 注意事项

1. **ONNX Runtime**：首次运行会自动下载ONNX Runtime CPU版本
2. **GPU加速**：如需GPU加速，需安装`onnxruntime-gpu`并配置CUDA环境
3. **模型路径**：确保模型文件路径正确
4. **图像格式**：支持JPEG、PNG、BMP格式

## 故障排除

### 问题1: ImportError: No module named 'onnxruntime'

```bash
pip install onnxruntime
```

### 问题2: PIL无法打开图像

```bash
pip install pillow
# 确保图像格式正确且路径无中文
```

### 问题3: CUDA版本不匹配

```bash
# 使用CPU版本
pip uninstall onnxruntime-gpu
pip install onnxruntime
```

## 下一步

- 使用真实图像进行测试
- 尝试不同模型
- 集成到自己的Python项目中
- 如需完整C++功能，请使用Qt6 Demo (`examples/qt6_demo/build/Release/VisionEngineDemo.exe`)
