# OTA模块CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 创建OTA库
add_library(vision_engine_ota STATIC
    ve_ota.cpp
    ve_ota_downloader.cpp
    ve_security.cpp
)

target_include_directories(vision_engine_ota PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vision_engine
)

target_link_libraries(vision_engine_ota PRIVATE
    vision_engine_core
)

install(TARGETS vision_engine_ota
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
