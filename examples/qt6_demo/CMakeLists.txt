cmake_minimum_required(VERSION 3.16)

project(VisionEngineDemo VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt6配置
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui Charts)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# VisionEngine库
find_package(VisionEngine REQUIRED)

# OpenCV (可选)
find_package(OpenCV QUIET)

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
    resources/resources.qrc
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${VisionEngine_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    Qt6::Charts
    VisionEngine::Core
)

# Windows特定配置
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        WIN32_EXECUTABLE ON
    )
endif()

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
