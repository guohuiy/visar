cmake_minimum_required(VERSION 3.16)
project(VisionEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 选项配置
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TOOLS "Build tools" ON)
option(ENABLE_ONNX "Enable ONNX Runtime backend" ON)
option(ENABLE_TENSORRT "Enable TensorRT backend" OFF)
option(ENABLE_NCNN "Enable NCNN backend" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)

# 第三方依赖
find_package(Threads REQUIRED)

# OpenCV (可选，用于图像处理)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: version ${OpenCV_VERSION}")
else()
    message(WARNING "OpenCV not found, some features may be disabled")
endif()

# ONNX Runtime
if(ENABLE_ONNX)
    find_package(ONNXRuntime QUIET)
    if(ONNXRuntime_FOUND)
        message(STATUS "ONNX Runtime found")
        set(HAVE_ONNX ON)
    else()
        message(WARNING "ONNX Runtime not found, set ONNXRuntime_DIR or disable ENABLE_ONNX")
        set(HAVE_ONNX OFF)
    endif()
endif()

# TensorRT
if(ENABLE_TENSORRT)
    find_package(TensorRT QUIET)
    if(TensorRT_FOUND)
        message(STATUS "TensorRT found")
        set(HAVE_TENSORRT ON)
    else()
        message(WARNING "TensorRT not found, set TensorRT_ROOT or disable ENABLE_TENSORRT")
        set(HAVE_TENSORRT OFF)
    endif()
endif()

# spdlog (日志库)
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(spdlog GIT_REPOSITORY https://github.com/gabime/spdlog.git
                        GIT_TAG v1.12.0)
    FetchContent_MakeAvailable(spdlog)
endif()

# 子目录
add_subdirectory(include/vision_engine)
add_subdirectory(src/core)
add_subdirectory(src/inference)
add_subdirectory(src/backends)
add_subdirectory(src/algorithms)
add_subdirectory(src/quantization)
add_subdirectory(src/ota)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples/qt6_demo)
endif()

if(BUILD_TOOLS)
    add_subdirectory(tools/model_converter)
endif()

# 打印配置摘要
message(STATUS "=== VisionEngine Build Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "Build Tools: ${BUILD_TOOLS}")
message(STATUS "ONNX Runtime: ${HAVE_ONNX}")
message(STATUS "TensorRT: ${HAVE_TENSORRT}")
message(STATUS "NCNN: ${HAVE_NCNN}")
message(STATUS "CUDA: ${ENABLE_CUDA}")
message(STATUS "OpenCV: ${OpenCV_FOUND}")
message(STATUS "=======================================")
