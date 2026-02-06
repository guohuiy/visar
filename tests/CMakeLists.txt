# Tests模块 CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 查找GoogleTest
find_package(GTest REQUIRED)

# 创建测试可执行文件
add_executable(vision_engine_tests
    test_main.cpp
    ve_model_test.cpp
    ve_inference_test.cpp
)

target_include_directories(vision_engine_tests PRIVATE
    ../../include/vision_engine
)

target_link_libraries(vision_engine_tests PRIVATE
    vision_engine_core
    vision_engine_inference
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(vision_engine_tests)

install(TARGETS vision_engine_tests
    RUNTIME DESTINATION bin
)
