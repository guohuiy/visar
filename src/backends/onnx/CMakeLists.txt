# ONNX Backend CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 检查是否启用ONNX
if(NOT HAVE_ONNX)
    return()
endif()

# 创建ONNX Runtime后端
add_library(ve_backend_onnx STATIC
    ve_onnx_backend.cpp
)

# 使用已设置的变量
target_include_directories(ve_backend_onnx PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vision_engine
    ${ONNXRuntime_INCLUDE_DIR}
)

target_link_libraries(ve_backend_onnx PRIVATE
    vision_engine_core
    ${ONNXRuntime_LIBRARY}
)

install(TARGETS ve_backend_onnx
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
