cmake_minimum_required(VERSION 3.16)

project(VisionEngineDemo VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows: 设置UTF-8编码
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
endif()

# Qt6配置 (MinGW)
# 注意：Qt6 MinGW缺少Qt6Config.cmake，禁用AUTOMOC
set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTORCC OFF)

# Qt6 MinGW路径
set(QT6_ROOT "C:/Qt/6.10.1/mingw_64")

# 使用自定义FindQt6模块（Qt6 MinGW缺少Qt6Config.cmake）
set(CMAKE_MODULE_PATH "C:/Users/huiya/Desktop" ${CMAKE_MODULE_PATH})
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui Charts)

# ONNX Runtime MinGW路径
set(ONNXRUNTIME_ROOT "C:/onnxruntime-win-x64-gpu-1.23.2" CACHE PATH "ONNX Runtime root")

# VisionEngine库路径（硬编码绝对路径）
set(VISION_ENGINE_ROOT "D:/codes/QT_V/visa")
set(VISION_ENGINE_INCLUDE_DIR "D:/codes/QT_V/visa/VisionEngine/include")
set(VISION_ENGINE_CORE_LIB "D:/codes/QT_V/visa/build/windows/src/core/Release/vision_engine_core.lib")

# ONNX Runtime库 (MinGW需要.a文件)
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
)

# MinGW链接静态库
if(MINGW)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Gui
        Qt6::Charts
        "${VISION_ENGINE_CORE_LIB}"
        -Wl,--start-group
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
        -Wl,--end-group
    )
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Gui
        Qt6::Charts
        "${VISION_ENGINE_CORE_LIB}"
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
    )
endif()

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
endif()

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
