# TensorRT Backend CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 检查是否启用TensorRT
if(NOT HAVE_TENSORRT)
    return()
endif()

# 创建TensorRT后端库
add_library(ve_backend_tensorrt STATIC
    ve_tensorrt_backend.cpp
)

# TensorRT头文件路径
set(TENSORRT_INCLUDE_DIR "" CACHE PATH "TensorRT include directory")
set(TENSORRT_LIBRARY_DIR "" CACHE PATH "TensorRT library directory")

# 查找TensorRT库
find_library(TENSORRT_LIBRARY 
    NAMES nvinfer nvonnxparser
    PATHS ${TENSORRT_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

# 设置包含目录
target_include_directories(ve_backend_tensorrt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vision_engine
    ${TENSORRT_INCLUDE_DIR}
)

# 链接库
target_link_libraries(ve_backend_tensorrt PRIVATE
    vision_engine_core
    ${TENSORRT_LIBRARY}
)

# 设置编译定义
target_compile_definitions(ve_backend_tensorrt PRIVATE
    HAVE_TENSORRT=1
)

install(TARGETS ve_backend_tensorrt
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
