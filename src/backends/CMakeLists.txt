# Backends模块CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# ONNX Runtime Backend
if(ENABLE_ONNX)
    add_subdirectory(onnx)
endif()

# TensorRT Backend
if(ENABLE_TENSORRT)
    add_subdirectory(tensorrt)
endif()

# NCNN Backend
if(ENABLE_NCNN)
    add_subdirectory(ncnn)
endif()
