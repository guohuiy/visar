#include <gtest/gtest.h>
#include "ve_inference.h"

TEST(InferenceEngineTest, CreateAndDestroy) {
    auto engine = std::make_unique<vision_engine::InferenceEngine>();
    EXPECT_NE(engine, nullptr);
}

TEST(InferenceEngineTest, Initialize) {
    vision_engine::InferenceEngine engine;
    vision_engine::EngineOptions options;
    auto status = engine.Initialize(options);
    EXPECT_EQ(status, VE_SUCCESS);
}

TEST(InferenceEngineTest, LoadInvalidModel) {
    vision_engine::InferenceEngine engine;
    auto status = engine.LoadModel("/nonexistent/model.onnx");
    EXPECT_EQ(status, VE_ERROR_FILE_NOT_FOUND);
}
