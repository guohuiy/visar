# Core模块CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 包含路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/vision_engine/core
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/vision_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/core
)

# 只创建静态库
add_library(vision_engine_core STATIC
    ve_logger.cpp
    ve_memory.cpp
    ve_options.cpp
    ve_thread_pool.cpp
)

target_link_libraries(vision_engine_core PRIVATE
    Threads::Threads
)

# 安装
install(TARGETS vision_engine_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
