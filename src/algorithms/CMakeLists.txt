# Algorithms模块CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 创建算法库
add_library(vision_engine_algorithms STATIC
    ve_algorithms.cpp
)

target_include_directories(vision_engine_algorithms PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(vision_engine_algorithms PRIVATE
    vision_engine_core
    vision_engine_inference
)

install(TARGETS vision_engine_algorithms
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
