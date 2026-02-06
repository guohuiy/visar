# Inference模块CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# ONNX Runtime (可选)
find_package(ONNXRuntime QUIET)
if(ONNXRuntime_FOUND)
    message(STATUS "ONNX Runtime found, enabling ONNX backend")
    add_definitions(-DHAVE_ONNX)
endif()

# 创建推理库
add_library(vision_engine_inference STATIC
    ve_model.cpp
    ve_result.cpp
    ve_inference.cpp
)

target_include_directories(vision_engine_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vision_engine
)

target_link_libraries(vision_engine_inference PRIVATE
    vision_engine_core
    ${ONNXRuntime_LIBRARIES}
)

install(TARGETS vision_engine_inference
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
