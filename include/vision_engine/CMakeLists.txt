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

# 导出配置
export(
    EXPORT VisionEngineTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/VisionEngineTargets.cmake"
    NAMESPACE VisionEngine::
)

# 创建Config文件
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/VisionEngineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/VisionEngineConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VisionEngine
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/VisionEngineConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    EXPORT VisionEngineTargets
    FILE VisionEngineTargets.cmake
    NAMESPACE VisionEngine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VisionEngine
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/VisionEngineConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/VisionEngineConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VisionEngine
)
