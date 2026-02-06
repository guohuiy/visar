# 头文件安装配置
cmake_minimum_required(VERSION 3.16)

# 定义VisionEngine_FOUND变量，供外部find_package使用
include(GNUInstallDirs)

# 创建配置头文件
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/vision_engine_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/vision_engine/vision_engine_config.h"
    @ONLY
)

# 安装头文件
install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/vision_engine
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)
