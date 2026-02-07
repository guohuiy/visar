# VisionEngine - 跨平台视觉推理引擎

## 项目概述

VisionEngine是一个跨平台的视觉推理引擎，支持多种推理后端（ONNX Runtime、TensorRT、NCNN），支持INT8量化和OTA热更新。

## 项目结构

```
VisionEngine/
├── CMakeLists.txt                    # 根CMake配置
├── README.md                         # 本文档
├── .gitignore                        # Git忽略文件
├── vision_engine_config.h.in         # 配置头文件模板
├── include/vision_engine/            # 公共头文件
│   ├── CMakeLists.txt
│   ├── vision_engine.h               # 主头文件
│   ├── core/                         # 核心接口层
│   │   ├── ve_types.h               # 类型定义
│   │   ├── ve_error.h               # 错误码定义
│   │   └── ve_options.h             # 配置选项
│   ├── inference/                    # 推理模块
│   │   ├── ve_inference.h           # 推理引擎接口
│   │   ├── ve_model.h               # 模型加载器
│   │   └── ve_result.h              # 推理结果
│   ├── quantization/                 # 量化模块
│   │   └── ve_quantization.h        # INT8量化引擎
│   ├── ota/                         # OTA模块
│   │   └── ve_ota.h                 # OTA更新器
│   ├── algorithms/                   # 算法模块
│   │   ├── ve_algorithms.h           # 目标检测/分割/OCR
│   │   ├── ve_nms.h                 # NMS后处理 (IoU/GIoU/DIoU/CIoU)
│   │   └── ve_detector.h            # YOLO/SSD输出解析器
│   └── backends/                    # 后端接口
│       ├── ve_onnx_backend.h         # ONNX Runtime后端
│       └── ve_tensorrt_backend.h    # TensorRT后端
├── src/                             # 源代码
│   ├── core/                         # 核心模块
│   │   ├── CMakeLists.txt
│   │   ├── ve_logger.cpp/h         # 日志系统
│   │   ├── ve_memory.cpp/h         # 内存管理
│   │   ├── ve_options.cpp          # 配置实现
│   │   ├── ve_engine.cpp           # 引擎主入口
│   │   └── ve_thread_pool.cpp/h    # 线程池
│   ├── inference/                    # 推理模块
│   │   ├── ve_inference.cpp        # 推理引擎实现
│   │   ├── ve_model.cpp            # 模型加载器实现
│   │   └── ve_result.cpp           # 推理结果实现
│   ├── backends/                     # 后端实现
│   │   ├── onnx/                   # ONNX Runtime后端
│   │   │   ├── ve_onnx_backend.cpp
│   │   │   └── CMakeLists.txt
│   │   └── tensorrt/               # TensorRT后端
│   │       ├── ve_tensorrt_backend.cpp
│   │       └── CMakeLists.txt
│   ├── algorithms/                   # 算法模块
│   ├── quantization/                 # 量化模块 (头文件已定义)
│   └── ota/                         # OTA模块 (头文件已定义)
├── examples/                         # 示例应用
│   ├── qt6_demo/                    # Qt6测试应用
│   └── python_demo/                 # Python测试示例
├── tests/                            # 测试
│   └── CMakeLists.txt
└── tools/                            # 工具
    └── model_converter/              # 模型转换工具
```

## 新增功能 (v2.0)

### 0. SIMD编译优化

```cmake
# CMake选项 (默认启用)
-DENABLE_AVX2=ON          # AVX2 SIMD加速
-DENABLE_AVX512=OFF       # AVX512 SIMD加速
-DENABLE_NEON=OFF         # ARM NEON SIMD加速

# 编译标志
# GCC/Clang: -mavx2 -mfma -O3 -ffast-math -funroll-loops
# MSVC: /arch:AVX2 /O2 /Oi /Ot /fp:fast
```

### 1. TensorRT GPU加速

支持NVIDIA GPU加速推理，提供FP16/INT8精度支持：

```cpp
#include "ve_tensorrt_backend.h"

using namespace vision_engine;

// 创建TensorRT后端
TensorRTBackend engine;

// 初始化 (支持 .onnx 或 .trt 文件)
VeEngineOptions options;
options.precision = VE_PRECISION_FP16;
engine.Initialize("model.trt", options);

// 启用FP16加速
engine.EnableFP16();

// 执行推理
std::vector<float> output_data(output_size);
engine.Infer(input_data.data(), input_shape, output_data.data());

// 预热模型
engine.Warmup(10);

// 获取GPU内存使用
size_t mem = engine.GetDeviceMemoryUsage();
```

### 2. NMS后处理模块

支持多种NMS算法：

```cpp
#include "ve_nms.h"

using namespace vision_engine::postprocess;

// 标准NMS
auto result = NonMaximumSuppression(detections, 0.45f);

// Soft-NMS (高斯衰减)
auto result = SoftNMS(detections, 0.45f, 0.5f, 1);

// CIoU-NMS (使用Complete IoU)
auto result = ClasswiseNMS(detections, 0.45f);

// 按类别执行NMS
auto result = ClasswiseNMS(detections, 0.45f);

// 置信度过滤
auto result = FilterByConfidence(detections, 0.5f);

// IoU计算
float iou = CalculateIoU(bbox_a, bbox_b);
float giou = CalculateGIoU(bbox_a, bbox_b);
float diou = CalculateDIoU(bbox_a, bbox_b);
float ciou = CalculateCIoU(bbox_a, bbox_b);
```

### 3. YOLO检测结果解析器

支持YOLOv5/v7/v8/vX输出解析：

```cpp
#include "ve_detector.h"

using namespace vision_engine::postprocess;

// 创建解析器
YOLOParser parser;
YOLOParser::Config config;
config.confidence_threshold = 0.5f;
config.nms_threshold = 0.45f;
config.num_classes = 80;
config.use_ciou = true;
parser.SetConfig(config);

// 解析YOLOv8输出
auto detections = parser.ParseYOLOv8(
    output_data.data(),
    output_shape,
    original_width,
    original_height,
    scale_x,
    scale_y,
    pad_x,
    pad_y
);

// 获取COCO类别名称
const char* name = GetCOCOClassName(class_id);

// 可视化检测结果
auto visualized = DetectionVisualizer::DrawDetections(
    image_data, width, height, detections);
```

### 4. 多尺度特征融合

支持多尺度特征图融合，提升小目标检测能力：

```cpp
#include "ve_multi_scale.h"

using namespace vision_engine::postprocess;

// 配置多尺度检测
MultiScaleDetector detector;
MultiScaleConfig config;
config.confidence_threshold = 0.5f;
config.nms_threshold = 0.45f;
config.strides = {8, 16, 32};  // P3, P4, P5
detector.SetConfig(config);

// 执行多尺度检测
auto detections = detector.Detect(outputs, shapes);
```

### 5. 快速NMS算法

使用空间哈希索引加速NMS，O(n)复杂度：

```cpp
#include "ve_fast_nms.h"

using namespace vision_engine::postprocess;

// 配置Fast-NMS
FastNMS nms;
FastNMSConfig config;
config.iou_threshold = 0.45f;
config.max_detections = 100;
config.grid_size = 64;
nms.SetConfig(config);

// 执行快速NMS
auto result = nms.Process(detections);
```

### 6. 内存池优化

预分配64字节对齐内存，减少malloc/free开销：

```cpp
#include "ve_memory_pool.h"

using namespace vision_engine;

// 获取预分配内存
auto& pool = TensorMemoryPool::Instance();
float* tensor = pool.AllocateTensor(640, 640, 3);

// 使用内存...

// 归还到池中
pool.Deallocate(tensor);
```

### 7. 测试时增强 (TTA)

通过多尺度、翻转等增强提升检测精度：

```cpp
#include "ve_tta.h"

using namespace vision_engine::tta;

// 配置TTA
TTAEngine tta;
TTAConfig config;
config.horizontal_flip = true;
config.multi_scale = true;
config.scales = {0.83f, 1.0f, 1.17f};
tta.SetConfig(config);

// 执行TTA推理
auto result = tta.Process(image, width, height, infer_func);
```

### 8. TensorRT INT8量化

支持TensorRT INT8量化推理：

```cpp
#include "ve_tensorrt_int8.h"

using namespace vision_engine;

// 配置INT8量化
TensorRTINT8Builder builder;
INT8QuantConfig config;
config.enable_int8 = true;
config.max_batch_size = 8;
builder.ConfigureINT8(config);

// 设置校准数据
builder.SetCalibrationData(calibration_data);

// 构建引擎
builder.BuildEngine("model_int8.trt");
```

### 9. Python量化工具

```bash
# PTQ量化
python tools/quantization/quantize_model.py \
    --model yolov8n.onnx \
    --calib-data ./calib_images/ \
    --output yolov8n_int8.onnx

# FP16量化
python tools/quantization/quantize_model.py \
    --model yolov8n.onnx \
    --quantize-type fp16 \
    --output yolov8n_fp16.onnx

# 带精度评估
python tools/quantization/quantize_model.py \
    --model yolov8n.onnx \
    --calib-data ./calib_images/ \
    --eval-data ./test_images/ \
    --output yolov8n_int8.onnx
```

## 性能优化预期

| 优化项 | 预期提升 |
|--------|----------|
| SIMD (AVX2) | 预处理速度 +40% |
| 内存池 | 推理延迟 -30% |
| Fast-NMS | NMS时间 -50% |
| 多尺度融合 | mAP +5-10% |
| TTA增强 | mAP +2-3% |
| INT8量化 | 速度 +2-4x, 模型 -4x |

## 编译状态

### Windows (MSVC) ✅ 已完成
- **输出文件**: 
  - `build/src/core/Release/vision_engine_core.lib`
  - `build/src/inference/Release/vision_engine_inference.lib`
  - `build/src/algorithms/Release/vision_engine_algorithms.lib`
- **编译器**: Visual Studio 2022 (MSVC 14.36.32548.0)

### Qt6 Demo ✅ 已完成
- **输出文件**: `examples/qt6_demo/build/Release/VisionEngineDemo.exe` (108 KB)
- **功能**: 基于Qt6的目标检测演示程序，支持图像加载和ONNX模型推理
- **模型路径**: `examples/qt6_demo/models/yolov8n.onnx`

### Linux ⏳ 待编译
- 使用 `build_linux.sh` 脚本在Linux环境下编译

### Python Demo ✅ 已完成
- **输出文件**: `examples/python_demo/test_vision_engine.py`
- **功能**: Python接口测试示例，支持ONNX模型推理测试
- **模型路径**: `examples/qt6_demo/models/yolov8n.onnx`
- **运行方式**:
  ```bash
  cd examples/python_demo
  pip install -r requirements.txt
  python test_vision_engine.py --model ../qt6_demo/models/yolov8n.onnx
  ```
- **测试结果**:
  - 模型: yolov8n.onnx (12.21 MB)
  - 推理时间: ~80ms (CPU)
  - 支持: 单图测试、批量测试、性能基准测试

## 快速开始

### Windows 编译 (MSVC)

```cmd
# 方法1: 使用编译脚本
build_windows.bat

# 方法2: 手动编译
mkdir build\windows
cd build\windows
cmake ..\VisionEngine -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### Windows 编译 (MinGW)

```cmd
build_windows_mingw.bat
```

### TensorRT 编译配置

```cmake
# CMake配置中启用TensorRT
-DENABLE_TENSORRT=ON
-DTENSORRT_INCLUDE_DIR="C:/TensorRT-10.10.0.31/include"
-DTENSORRT_LIBRARY_DIR="C:/TensorRT-10.10.0.31/lib"
```

## 接口使用说明

### C++ 接口

#### 1. 创建推理引擎

```cpp
#include "vision_engine.h"
#include "ve_inference.h"
#include "ve_model.h"
#include "ve_result.h"

using namespace vision_engine;

// 配置引擎选项
VeEngineOptions options;
options.preferred_backend = VE_BACKEND_ONNX;  // 选择后端
options.device_type = VE_DEVICE_CUDA;         // 选择设备
options.precision = VE_PRECISION_FP32;       // 选择精度
options.num_threads = 4;                      // 线程数
options.batch_size = 1;                       // 批大小

// 创建推理引擎
auto engine = std::make_shared<InferenceEngine>();
VeStatusCode status = engine->Initialize(options);

if (status != VE_SUCCESS) {
    // 处理错误
    printf("引擎初始化失败: %s\n", ve_status_string(status));
}
```

#### 2. 加载模型

```cpp
// 加载ONNX模型
VeStatusCode status = engine->LoadModel("path/to/model.onnx");

if (status != VE_SUCCESS) {
    // 处理错误
    printf("模型加载失败: %s\n", ve_status_string(status));
}

// 获取模型信息
VeModelInfo model_info = engine->GetModelInfo();
printf("模型名称: %s\n", model_info.name);
printf("输入尺寸: %d x %d\n", model_info.input_width, model_info.input_height);
```

#### 3. 图像推理

```cpp
// 准备图像数据
VeImageData image;
image.data = image_buffer;           // 图像数据指针
image.width = 640;                   // 图像宽度
image.height = 480;                  // 图像高度
image.format = VE_IMAGE_FORMAT_RGB;   // 图像格式

// 设置归一化参数 (可选)
image.mean = new float[3]{123.675, 116.28, 103.53};  // RGB均值
image.std = new float[3]{58.395, 57.12, 57.375};      // RGB标准差

// 执行推理
auto result = engine->Infer(image);

// 处理检测结果
int32_t count = result->GetDetectionCount();
printf("检测到 %d 个目标:\n", count);

for (int32_t i = 0; i < count; i++) {
    VeDetection det = result->GetDetection(i);
    printf("  [%d] 类别: %d, 置信度: %.2f\n", i, det.class_id, det.score);
    printf("       位置: (%.1f, %.1f) - (%.1f, %.1f)\n",
           det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2);
}

// 获取推理时间
double inference_time = result->GetInferenceTimeMs();
printf("推理耗时: %.2f ms\n", inference_time);

// 获取JSON格式结果
std::string json_result = result->ToJSON();
```

#### 4. 异步推理

```cpp
// 设置结果回调
engine->SetResultCallback([](std::shared_ptr<InferenceResult> result) {
    printf("异步推理完成，检测到 %d 个目标\n", result->GetDetectionCount());
});

// 发起异步推理
auto future = engine->InferAsync(image);

// 等待结果
auto result = future.get();
```

#### 5. 批量推理

```cpp
// 准备批量图像
std::vector<VeImageData> images;
for (int i = 0; i < batch_size; i++) {
    VeImageData img;
    // 设置每个图像...
    images.push_back(img);
}

// 执行批量推理
auto batch_result = engine->InferBatch(images.data(), batch_size);

// 获取批量结果
for (int i = 0; i < batch_result->GetBatchSize(); i++) {
    auto result = batch_result->GetResult(i);
    printf("批次[%d]: 检测到 %d 个目标\n", i, result->GetDetectionCount());
}

printf("平均推理时间: %.2f ms\n", batch_result->GetAverageInferenceTimeMs());
```

#### 6. 配置参数

```cpp
// 设置置信度阈值 (默认0.5)
engine->SetConfidenceThreshold(0.7f);

// 设置NMS阈值 (默认0.45)
engine->SetNMSThreshold(0.5f);

// 预热模型
engine->Warmup();

// 获取支持的设备
auto devices = engine->GetSupportedDevices();
for (auto device : devices) {
    printf("支持设备: %d\n", device);
}

// 获取支持的精度
auto precisions = engine->GetSupportedPrecisions(VE_DEVICE_CUDA);
for (auto precision : precisions) {
    printf("支持精度: %d\n", precision);
}
```

#### 7. 模型管理

```cpp
// 创建模型管理器
ModelManager manager;
manager.SetModelPath("path/to/models");

// 注册模型
manager.RegisterModel("yolov8", "models/yolov8.onnx");
manager.RegisterModel("resnet50", "models/resnet50.onnx");

// 列出已注册模型
auto models = manager.ListModels();
for (const auto& name : models) {
    printf("模型: %s\n", name.c_str());
}

// 获取模型
auto model = manager.GetModel("yolov8");
```

### C 接口

#### 1. 创建和销毁引擎

```c
// 创建引擎
VeInferenceHandle handle = ve_inference_create(&options);

if (handle == NULL) {
    printf("引擎创建失败\n");
    return;
}

// 使用引擎...

// 销毁引擎
ve_inference_destroy(handle);
```

#### 2. 加载和推理

```c
// 加载模型
VeStatusCode status = ve_inference_load_model(handle, "model.onnx");

// 推理
VeImageData image;
image.data = image_buffer;
image.width = 640;
image.height = 480;
image.format = VE_IMAGE_FORMAT_RGB;

VeInferenceResult* result = ve_inference_infer(handle, &image);

// 处理结果
int32_t count = ve_result_get_detection_count(result);
float time = ve_result_get_inference_time(result);

printf("检测到 %d 个目标，耗时 %.2f ms\n", count, time);

// 释放结果
ve_result_destroy(result);
```

#### 3. 异步推理

```c
void inference_callback(VeInferenceResult* result) {
    int32_t count = ve_result_get_detection_count(result);
    printf("异步推理完成: %d 个目标\n", count);
    ve_result_destroy(result);
}

// 设置回调
ve_inference_set_callback(handle, inference_callback);

// 发起异步推理
ve_inference_infer_async(handle, &image, inference_callback);
```

#### 4. 批量推理

```c
VeInferenceResult** results;
int32_t result_count;

results = ve_inference_infer_batch(handle, images, batch_size, &result_count);

for (int32_t i = 0; i < result_count; i++) {
    int32_t count = ve_result_get_detection_count(results[i]);
    printf("批次[%d]: %d 个目标\n", i, count);
    ve_result_destroy(results[i]);
}

free(results);
```

### Python 接口

Python测试示例位于 `examples/python_demo/`，提供以下功能：

#### 1. 安装依赖

```bash
cd examples/python_demo
pip install -r requirements.txt
```

#### 2. 运行测试

```bash
# 默认测试
python test_vision_engine.py

# 指定模型测试
python test_vision_engine.py --model ../qt6_demo/models/yolov8n.onnx
```

#### 3. Python API 示例

```python
from test_vision_engine import VisionEngineSimulator, test_single_image

# 创建引擎并加载模型
engine = VisionEngineSimulator("path/to/model.onnx")

# 测试单张图像
detections = test_single_image(engine, "path/to/image.jpg")

# 查看检测结果
for det in detections:
    print(f"类别: {det['class_name']}, 置信度: {det['score']:.4f}")
    bbox = det['bbox']
    print(f"位置: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) - ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
```

## 常用类型说明

```c
// 设备类型
typedef enum {
    VE_DEVICE_CPU = 0,      // CPU
    VE_DEVICE_GPU,          // GPU (通用)
    VE_DEVICE_CUDA,         // NVIDIA CUDA
    VE_DEVICE_VULKAN,       // Vulkan
    VE_DEVICE_OPENCL,       // OpenCL
    VE_DEVICE_NPU          // 神经网络处理器
} VeDeviceType;

// 数据精度
typedef enum {
    VE_PRECISION_FP32 = 0,  // 32位浮点
    VE_PRECISION_FP16,      // 16位浮点
    VE_PRECISION_BF16,      // Brain Float16
    VE_PRECISION_INT8,      // 8位整数 (量化)
    VE_PRECISION_INT4,      // 4位整数
} VePrecision;

// 推理后端
typedef enum {
    VE_BACKEND_ONNX = 0,    // ONNX Runtime
    VE_BACKEND_TENSORRT,   // NVIDIA TensorRT
    VE_BACKEND_OPENVINO,   // Intel OpenVINO
    VE_BACKEND_NCNN,       // 腾讯NCNN
} VeBackendType;

// 图像格式
typedef enum {
    VE_IMAGE_FORMAT_RGB = 0,    // RGB
    VE_IMAGE_FORMAT_BGR,         // BGR
    VE_IMAGE_FORMAT_RGBA,        // RGBA
    VE_IMAGE_FORMAT_GRAY,       // 灰度
} VeImageFormat;

// 检测结果
typedef struct {
    float x1, y1;           // 左上角坐标
    float x2