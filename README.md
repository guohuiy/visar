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
│   │   └── ve_algorithms.h         # 目标检测/分割/OCR
│   └── backends/                    # 后端接口
│       ├── ve_onnx_backend.h        # ONNX Runtime后端
│       └── ve_tensorrt_backend.h   # TensorRT后端
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
│   │   └── tensorrt/               # TensorRT后端
│   ├── algorithms/                   # 算法模块 (待开发)
│   ├── quantization/                 # 量化模块 (头文件已定义)
│   └── ota/                         # OTA模块 (头文件已定义)
├── examples/                         # 示例应用
│   └── qt6_demo/                    # Qt6测试应用 (待开发)
├── tests/                            # 测试
│   └── CMakeLists.txt
└── tools/                            # 工具
    └── model_converter/              # 模型转换工具 (待开发)
```

## 编译状态

### Windows (MSVC) ✅ 已完成
- **输出文件**: 
  - `build/windows/src/core/Release/vision_engine_core.lib` (234 KB)
  - `build/windows/src/inference/Release/vision_engine_inference.lib` (869 KB)
- **编译器**: Visual Studio 2022 (MSVC 14.36.32548.0)

### Linux ⏳ 待编译
- 使用 `build_linux.sh` 脚本在Linux环境下编译

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

### 常用类型说明

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
    float x2, y2;           // 右下角坐标
    float score;            // 置信度
    int32_t class_id;       // 类别ID
    const char* class_name; // 类别名称
} VeDetection;

// 性能指标
typedef struct {
    double preprocess_time_ms;   // 预处理时间
    double inference_time_ms;    // 推理时间
    double postprocess_time_ms;  // 后处理时间
    double total_time_ms;        // 总时间
    size_t memory_used_bytes;   // 内存使用量
} VePerformanceMetrics;
```

## 开发状态

### 已完成 ✅
- [x] 项目架构设计
- [x] 核心类型定义
- [x] 错误处理机制
- [x] 日志系统
- [x] 内存管理
- [x] 线程池
- [x] 引擎主入口
- [x] CMake构建系统
- [x] Windows编译
- [x] 推理模块实现 (inference/)
- [x] ONNX Runtime后端集成
- [x] TensorRT后端集成

### 待开发 ⏳
- [ ] INT8量化引擎
- [ ] OTA热更新系统
- [ ] 算法模块 (YOLO/OCR)
- [ ] Qt6测试Demo
- [ ] Linux跨平台测试

## 依赖库

| 依赖 | 路径 | 状态 |
|------|------|------|
| ONNX Runtime GPU | C:\onnxruntime-win-x64-gpu-1.23.2 | 已集成 |
| TensorRT | C:\TensorRT-10.10.0.31 | 已集成 |
| OpenCV | C:\opencv\ | 可选 |
| Qt6 | C:\Qt\ | 可选 |
| NCNN | D:\tools\ncnn | 可选 |

## GitHub

项目地址: https://github.com/guohuiy/visar.git

## 许可证

MIT License
