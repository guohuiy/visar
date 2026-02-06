# Quantization模块CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 创建量化库
add_library(vision_engine_quantization STATIC
    ve_quantization.cpp
)

target_include_directories(vision_engine_quantization PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vision_engine
)

target_link_libraries(vision_engine_quantization PRIVATE
    vision_engine_core
)

install(TARGETS vision_engine_quantization
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
