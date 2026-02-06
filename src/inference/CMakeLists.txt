# Inference模块CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 包含路径 - 添加所有子目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/vision_engine/core
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/vision_engine/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/vision_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/inference
)

# 创建推理库
add_library(vision_engine_inference STATIC
    ve_model.cpp
    ve_result.cpp
    ve_inference.cpp
)

if(HAVE_ONNX)
    target_include_directories(vision_engine_inference PRIVATE
        ${ONNXRuntime_INCLUDE_DIR}
    )
    target_link_libraries(vision_engine_inference PRIVATE
        ${ONNXRuntime_LIBRARY}
    )
endif()

install(TARGETS vision_engine_inference
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
