cmake_minimum_required(VERSION 3.16)
project(VisionEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 全局包含路径
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /source-charset:utf-8")

# 添加自定义模块路径
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 全局包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 选项配置 - 禁用复杂模块
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_TOOLS "Build tools" OFF)
option(ENABLE_ONNX "Enable ONNX Runtime backend" ON)
option(ENABLE_TENSORRT "Enable TensorRT backend" OFF)
option(ENABLE_NCNN "Enable NCNN backend" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)

# 第三方依赖
find_package(Threads REQUIRED)

# ONNX Runtime - 硬编码Windows路径
if(ENABLE_ONNX)
    if(WIN32)
        set(ONNXRuntime_ROOT "C:/onnxruntime-win-x64-gpu-1.23.2" CACHE PATH "ONNX Runtime root")
        set(ONNXRuntime_INCLUDE_DIR "${ONNXRuntime_ROOT}/include")
        set(ONNXRuntime_LIBRARY "${ONNXRuntime_ROOT}/lib/onnxruntime.lib")
        set(ONNXRuntime_FOUND TRUE)
    else()
        find_package(ONNXRuntime REQUIRED)
    endif()
    
    if(ONNXRuntime_FOUND)
        message(STATUS "ONNX Runtime found: ${ONNXRuntime_INCLUDE_DIR}")
        set(HAVE_ONNX ON)
        add_definitions(-DHAVE_ONNX)
    else()
        message(FATAL_ERROR "ONNX Runtime not found")
    endif()
endif()

# TensorRT
if(ENABLE_TENSORRT)
    set(TensorRT_ROOT "" CACHE PATH "TensorRT root directory")
    find_package(TensorRT QUIET)
    if(TensorRT_FOUND)
        message(STATUS "TensorRT found")
        set(HAVE_TENSORRT ON)
        add_definitions(-DHAVE_TENSORRT)
    else()
        message(WARNING "TensorRT not found")
        set(HAVE_TENSORRT OFF)
    endif()
endif()

# 日志系统 - 使用内联实现，无外部依赖
message(STATUS "Using inline logger implementation")

# 只编译核心模块
add_subdirectory(include/vision_engine)
add_subdirectory(src/core)

# 子目录 - 禁用inference等复杂模块
# add_subdirectory(src/inference)
# add_subdirectory(src/backends)
# add_subdirectory(src/algorithms)
# add_subdirectory(src/quantization)
# add_subdirectory(src/ota)

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
message(STATUS "OpenCV: OFF")
message(STATUS "=======================================")
