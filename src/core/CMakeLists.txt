# Core模块CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 创建vision_engine_core库
add_library(vision_engine_core STATIC
    ve_logger.cpp
    ve_memory.cpp
    ve_options.cpp
    ve_thread_pool.cpp
)

target_include_directories(vision_engine_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vision_engine
)

target_link_libraries(vision_engine_core PRIVATE
    spdlog::spdlog
    Threads::Threads
)

# 安装
install(TARGETS vision_engine_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# 创建动态库
add_library(vision_engine SHARED
    ve_logger.cpp
    ve_memory.cpp
    ve_options.cpp
    ve_thread_pool.cpp
    ve_engine.cpp
)

target_include_directories(vision_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vision_engine
)

target_link_libraries(vision_engine PRIVATE
    vision_engine_core
    spdlog::spdlog
    Threads::Threads
)

if(WIN32)
    set_target_properties(vision_engine PROPERTIES
        PREFIX ""  # 移除lib前缀
    )
endif()

install(TARGETS vision_engine
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
