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

### 后处理
- YOLOv8输出解析
- NMS非极大值抑制
- 边界框坐标转换
- 类别名称映射 (COCO 80类)

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
└── models/                # 模型目录 (可选)
    └── yolov8n.onnx
```

## 与C++ API对应关系

| Python函数 | C++ API | 说明 |
|-----------|---------|------|
| `VisionEngineSimulator()` | `InferenceEngine()` | 创建推理引擎 |
| `load_model()` | `LoadModel()` | 加载模型 |
| `infer()` | `Infer()` | 执行推理 |
| `SetConfidenceThreshold()` | `SetConfidenceThreshold()` | 设置置信度阈值 |
| `SetNMSThreshold()` | `SetNMSThreshold()` | 设置NMS阈值 |

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
