# 模型转换工具 CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

project(VisionEngineTools VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(model_converter
    main.cpp
    onnx_converter.cpp
    quantize_tool.cpp
)

target_include_directories(model_converter PRIVATE
    ../../include/vision_engine
)

target_link_libraries(model_converter PRIVATE
    vision_engine_core
)

install(TARGETS model_converter
    RUNTIME DESTINATION bin
)
