cmake_minimum_required(VERSION 3.16)

project(VisionEngineDemo VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows: 设置UTF-8编码
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
endif()

# Qt6配置 (MSVC) - 启用MOC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC OFF)

# Qt6 MSVC路径
set(QT6_ROOT "C:/Qt/6.10.1/msvc2022_64")
set(CMAKE_PREFIX_PATH "${QT6_ROOT}")

# 查找Qt6
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui Charts)

# ONNX Runtime路径
set(ONNXRUNTIME_ROOT "C:/onnxruntime-win-x64-gpu-1.23.2" CACHE PATH "ONNX Runtime root")

# VisionEngine库路径
set(VISION_ENGINE_ROOT "D:/codes/QT_V/visar-master")
set(VISION_ENGINE_INCLUDE_DIR "D:/codes/QT_V/visar-master/VisionEngine/include")
set(VISION_ENGINE_CORE_LIB "D:/codes/QT_V/visar-master/build/windows/src/core/Release/vision_engine_core.lib")

# ONNX Runtime库
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")

# 源文件
set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/InferenceWorker.cpp
    src/ModelManager.cpp
    src/OTADialog.cpp
    src/PerformanceMonitor.cpp
)

set(HEADERS
    include/MainWindow.h
    include/InferenceWorker.h
    include/ModelManager.h
    include/OTADialog.h
    include/PerformanceMonitor.h
)

# 创建可执行文件
add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${VISION_ENGINE_INCLUDE_DIR}
    ${VISION_ENGINE_INCLUDE_DIR}/vision_engine
    ${VISION_ENGINE_INCLUDE_DIR}/vision_engine/core
    "${ONNXRUNTIME_ROOT}/include"
    "${QT6_ROOT}/include"
)

# 链接Qt6库
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    Qt6::Charts
    "${VISION_ENGINE_CORE_LIB}"
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
)

# Windows特定配置
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        WIN32_EXECUTABLE ON
    )
    
    # 复制ONNX Runtime DLL到输出目录
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/onnxruntime.dll"
        COMMENT "Copying ONNX Runtime DLL"
    )
    
    # 复制Qt DLL到输出目录（MSVC Release版本）
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${QT6_ROOT}/bin/Qt6Core.dll"
            "${QT6_ROOT}/bin/Qt6Gui.dll"
            "${QT6_ROOT}/bin/Qt6Widgets.dll"
            "${QT6_ROOT}/bin/Qt6Charts.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/"
        COMMENT "Copying Qt DLLs"
    )
    
    # 复制Qt platforms插件（解决 no Qt platform plugin 错误）
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/platforms"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${QT6_ROOT}/plugins/platforms/qwindows.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/platforms/"
        COMMENT "Copying Qt platform plugin"
    )
    
    # 复制Qt依赖的翻译文件
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/translations"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${QT6_ROOT}/translations/qtbase_*.qm"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/translations/"
        COMMENT "Copying Qt translations"
    )
endif()

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
