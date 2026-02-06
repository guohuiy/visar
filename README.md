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
│       └── ve_onnx_backend.h        # ONNX Runtime后端
├── src/                             # 源代码
│   ├── core/                         # 核心模块
│   │   ├── CMakeLists.txt
│   │   ├── ve_logger.cpp/h         # 日志系统
│   │   ├── ve_memory.cpp/h         # 内存管理
│   │   ├── ve_options.cpp          # 配置实现
│   │   ├── ve_engine.cpp           # 引擎主入口
│   │   └── ve_thread_pool.cpp/h    # 线程池
│   ├── inference/                    # 推理模块 (待实现)
│   ├── backends/                     # 后端实现 (待开发)
│   ├── algorithms/                   # 算法模块 (待开发)
│   ├── quantization/                 # 量化模块 (头文件已定义)
│   └── ota/                         # OTA模块 (头文件已定义)
├── examples/                         # 示例应用
│   └── qt6_demo/                    # Qt6测试应用 (待开发)
├── tests/                            # 测试 (待开发)
│   └── CMakeLists.txt
└── tools/                            # 工具
    └── model_converter/              # 模型转换工具 (待开发)
```

## 编译状态

### Windows (MSVC) ✅ 已完成
- **输出文件**: `build/windows/src/core/Release/vision_engine_core.lib`
- **文件大小**: 234 KB
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

### 待开发 ⏳
- [ ] 推理模块实现 (inference/)
- [ ] ONNX Runtime后端集成
- [ ] TensorRT后端集成
- [ ] INT8量化引擎
- [ ] OTA热更新系统
- [ ] 算法模块 (YOLO/OCR)
- [ ] Qt6测试Demo
- [ ] Linux跨平台测试

## 依赖库

| 依赖 | 路径 | 状态 |
|------|------|------|
| ONNX Runtime GPU | C:\onnxruntime-win-x64-gpu-1.23.2 | 已配置 |
| TensorRT | C:\TensorRT-10.10.0.31 | 待集成 |
| OpenCV | C:\opencv\ | 可选 |
| Qt6 | C:\Qt\ | 可选 |
| NCNN | D:\tools\ncnn | 可选 |

## GitHub

项目地址: https://github.com/guohuiy/visar.git

## 许可证

MIT License
