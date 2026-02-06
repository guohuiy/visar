# VisionEngine - 跨平台视觉推理引擎

## 项目概述

VisionEngine是一个跨平台的视觉推理引擎，支持多种推理后端（ONNX Runtime、TensorRT、NCNN），支持INT8量化和OTA热更新。根据visa.md文档设计实现。

## 项目结构

```
VisionEngine/
├── CMakeLists.txt                    # 根CMake配置
├── README.md                         # 本文档
├── .gitignore                        # Git忽略文件
├── include/vision_engine/            # 公共头文件
│   ├── CMakeLists.txt
│   ├── vision_engine.h               # 主头文件
│   ├── vision_engine_config.h.in
│   ├── VisionEngineConfig.cmake.in
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
│   ├── algorithms/                   # 算法模块 (新增)
│   │   └── ve_algorithms.h          # 目标检测/分割/OCR
│   └── backends/                    # 后端接口
│       └── ve_onnx_backend.h        # ONNX Runtime后端
├── src/                             # 源代码
│   ├── core/                         # 核心模块
│   │   ├── CMakeLists.txt
│   │   ├── ve_logger.cpp/h          # 日志系统
│   │   ├── ve_memory.cpp/h          # 内存管理
│   │   ├── ve_options.cpp           # 配置实现
│   │   ├── ve_engine.cpp            # 引擎主入口
│   │   └── ve_thread_pool.cpp/h     # 线程池
│   ├── inference/                    # 推理模块
│   │   ├── CMakeLists.txt
│   │   ├── ve_model.cpp             # 模型加载
│   │   ├── ve_result.cpp            # 结果处理
│   │   └── ve_inference.cpp         # 推理引擎
│   ├── backends/                     # 后端实现
│   │   ├── CMakeLists.txt
│   │   └── onnx/                    # ONNX后端
│   │       ├── CMakeLists.txt
│   │       └── ve_onnx_backend.cpp
│   ├── algorithms/                   # 算法模块 (新增)
│   │   ├── CMakeLists.txt
│   │   └── ve_algorithms.cpp        # YOLO/UNet/OCR实现
│   ├── quantization/                 # 量化模块
│   │   ├── CMakeLists.txt
│   │   └── ve_quantization.cpp      # INT8量化实现
│   └── ota/                         # OTA模块
│       ├── CMakeLists.txt
│       ├── ve_ota.cpp               # OTA更新
│       ├── ve_security.cpp           # 安全验证
│       └── ve_ota_downloader.cpp     # 下载器
├── examples/                         # 示例应用
│   └── qt6_demo/                    # Qt6测试应用
│       ├── CMakeLists.txt
│       ├── include/
│       │   ├── MainWindow.h         # 主窗口
│       │   ├── InferenceWorker.h    # 后台推理线程
│       │   ├── ModelManager.h       # 模型管理
│       │   ├── OTADialog.h         # OTA对话框
│       │   └── PerformanceMonitor.h  # 性能监控
│       ├── src/
│       │   ├── main.cpp
│       │   ├── MainWindow.cpp       # 主窗口实现
│       │   ├── InferenceWorker.cpp   # 推理工作线程
│       │   ├── ModelManager.cpp     # 模型管理
│       │   ├── OTADialog.cpp        # OTA对话框
│       │   └── PerformanceMonitor.cpp
│       └── resources/
│           └── resources.qrc         # Qt资源文件
├── tools/                            # 工具
│   └── model_converter/              # 模型转换工具
│       ├── CMakeLists.txt
│       ├── main.cpp
│       ├── onnx_converter.cpp       # ONNX转换
│       └── quantize_tool.cpp         # 量化工具
└── tests/                            # 测试
    ├── CMakeLists.txt
    ├── test_main.cpp
    ├── ve_model_test.cpp
    └── ve_inference_test.cpp
```

## 架构设计

### 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                      UI层 (应用层)                           │
│          (手机App/PC客户端/Web界面等)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │   C API 接口层        │
                │  (标准C接口导出)      │
                └───────────┬───────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                  核心引擎层 (C/C++)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 推理管理器   │  │  模型加载器  │  │  内存管理器  │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
│  ┌────────────────────────────────────────────────────┐    │
│  │        算法模块 (可插拔设计)                       │    │
│  │  - 目标检测 (YOLO/RCNN系列)                      │    │
│  │  - 图像分割 (Semantic/Instance Segmentation)     │    │
│  │  - OCR识别 (文本检测+识别)                        │    │
│  └────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              推理后端层 (Runtime Backends)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────────┐     │
│  │ ONNX RT  │ │ TensorRT │ │ OpenVINO│ │ NCNN/MNN   │     │
│  │ (通用)   │ │ (NVIDIA) │ │ (Intel) │ │ (移动端)   │     │
│  └──────────┘ └──────────┘ └─────────┘ └────────────┘     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              硬件抽象层 (HAL)                                │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌────────────┐              │
│  │  CUDA    │ │ OpenCL   │ │   ARM      │              │
│  │  (GPU)   │ │ (跨平台) │ │ (移动芯片) │              │
│  └──────────┘ └──────────┘ └────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## 编译方法

### Windows (Visual Studio 2022)

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -DENABLE_CUDA=ON
cmake --build . --config Release
```

### Linux

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
make -j$(nproc)
```

### macOS

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 依赖项

- CMake 3.16+
- C++17 编译器
- spdlog (日志库)
- ONNX Runtime 1.16+ (可选)
- TensorRT 8.6+ (可选，NVIDIA GPU)
- NCNN (可选，移动端)
- Qt6 6.0+ (用于Qt6 Demo)
- OpenCV 4.8+ (可选)

## Qt6 Demo功能

- **图像加载与显示**: 支持PNG、JPG、BMP格式
- **视频加载**: 支持视频文件推理
- **摄像头实时推理**: 实时视频流处理
- **模型选择**: YOLOv5s, ResNet50, MobileNet-SSD
- **后端选择**: ONNX Runtime (CPU/CUDA), TensorRT
- **量化选项**: FP32, FP16, INT8
- **OTA模型更新检查**: 检查并下载新模型
- **性能监控**: 推理时间统计（Avg/Min/Max）
- **日志显示**: 实时推理日志

## 主要特性

1. **多后端支持**: ONNX Runtime, TensorRT, OpenVINO, NCNN
2. **INT8量化引擎**: PTQ后训练量化、动态量化
3. **OTA热更新**: 模型无缝切换、增量更新
4. **算法模块**: YOLO目标检测、UNet/DeepLab分割、OCR识别
5. **C API接口**: 支持多语言绑定
6. **跨平台设计**: Windows, Linux, macOS, Android, iOS

## 开发路线图

| Phase | 内容 | 时间 |
|-------|------|------|
| Phase 1 | 核心框架 + ONNX Runtime集成 | 2个月 |
| Phase 2 | TensorRT/NCNN后端开发 | 1.5个月 |
| Phase 3 | 性能优化 + 多线程调度 | 1个月 |
| Phase 4 | 跨平台测试 + API封装 | 0.5个月 |

## 性能指标

| 指标 | 目标值 |
|------|--------|
| 推理延迟 | < 50ms (移动端), < 10ms (PC GPU) |
| 吞吐量 | > 30 FPS (移动端), > 100 FPS (PC) |
| 内存占用 | < 200MB (移动端), < 1GB (PC) |
| INT8加速比 | > 2.5x FP32 |

## 版本历史

- v1.0.0 (2024): 初始版本 - 基础架构、ONNX Runtime集成、Qt6 Demo
