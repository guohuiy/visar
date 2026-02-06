#include <gtest/gtest.h>
#include "ve_model.h"

TEST(ModelLoaderTest, LoadInvalidPath) {
    vision_engine::ModelLoader loader;
    auto status = loader.LoadModel("/nonexistent/model.onnx");
    EXPECT_EQ(status, VE_ERROR_FILE_NOT_FOUND);
}

TEST(ModelLoaderTest, CreateAndDestroy) {
    auto loader = std::make_unique<vision_engine::ModelLoader>();
    EXPECT_NE(loader, nullptr);
}
