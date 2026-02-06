# ONNX Backend CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 创建ONNX Runtime后端
add_library(ve_backend_onnx STATIC
    ve_onnx_backend.cpp
)

find_package(ONNXRuntime REQUIRED)

target_include_directories(ve_backend_onnx PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vision_engine
)

target_link_libraries(ve_backend_onnx PRIVATE
    vision_engine_core
    onnxruntime::onnxruntime
)

install(TARGETS ve_backend_onnx
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
