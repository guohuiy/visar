#include "ve_inference.h"
#include "ve_result.h"
#include "ve_model.h"
#include "../core/ve_logger.h"
#include <fstream>
#include <algorithm>

namespace vision_engine {

class InferenceEngine::Impl {
public:
    VeEngineOptions options_;
    bool model_loaded_ = false;
    std::string last_error_;
    VeModelInfo model_info_;
    ResultCallback result_callback_;
    float confidence_threshold_ = 0.5f;
    float nms_threshold_ = 0.45f;
};

InferenceEngine::InferenceEngine() : impl_(std::make_unique<Impl>()) {}

InferenceEngine::~InferenceEngine() {
    UnloadModel();
}

VeStatusCode InferenceEngine::Initialize(const VeEngineOptions& options) {
    impl_->options_ = options;
    VE_LOG_INFO("InferenceEngine initialized");
    return VE_SUCCESS;
}

VeStatusCode InferenceEngine::LoadModel(const std::string& model_path) {
    VE_LOG_INFO("Loading model: " + model_path);
    
    // 检查文件是否存在
    std::ifstream file(model_path);
    if (!file.is_open()) {
        impl_->last_error_ = "Model file not found: " + model_path;
        return VE_ERROR_FILE_NOT_FOUND;
    }
    
    impl_->model_loaded_ = true;
    impl_->model_info_.path = model_path.c_str();
    impl_->model_info_.name = "Loaded Model";
    
    VE_LOG_INFO("Model loaded successfully");
    return VE_SUCCESS;
}

void InferenceEngine::UnloadModel() {
    impl_->model_loaded_ = false;
}

bool InferenceEngine::IsModelLoaded() const {
    return impl_->model_loaded_;
}

std::shared_ptr<InferenceResult> InferenceEngine::Infer(const VeImageData& image) {
    auto result = std::make_shared<InferenceResult>();
    
    if (!impl_->model_loaded_) {
        impl_->last_error_ = "Model not loaded";
        return result;
    }
    
    // 模拟推理时间
    impl_->model_info_.input_width = image.width;
    impl_->model_info_.input_height = image.height;
    
    return result;
}

std::shared_ptr<InferenceResult> InferenceEngine::Infer(const uint8_t* data,
                                                          int32_t width,
                                                          int32_t height,
                                                          VeImageFormat format) {
    VeImageData image;
    image.data = const_cast<uint8_t*>(data);
    image.width = width;
    image.height = height;
    image.format = format;
    return Infer(image);
}

std::future<std::shared_ptr<InferenceResult>> InferenceEngine::InferAsync(const VeImageData& image) {
    return std::async(std::launch::async, [this, image]() {
        return Infer(image);
    });
}

void InferenceEngine::SetResultCallback(ResultCallback callback) {
    impl_->result_callback_ = callback;
}

std::shared_ptr<BatchResult> InferenceEngine::InferBatch(const VeImageData* images, 
                                                          int32_t batch_size) {
    auto batch_result = std::make_shared<BatchResult>();
    for (int32_t i = 0; i < batch_size; ++i) {
        auto result = Infer(images[i]);
        batch_result->AddResult(result);
    }
    return batch_result;
}

VeStatusCode InferenceEngine::Warmup() {
    VE_LOG_INFO("Warming up model...");
    return VE_SUCCESS;
}

VeModelInfo InferenceEngine::GetModelInfo() const {
    return impl_->model_info_;
}

VePerformanceMetrics InferenceEngine::GetPerformanceMetrics() const {
    return VePerformanceMetrics{};
}

void InferenceEngine::SetConfidenceThreshold(float threshold) {
    impl_->confidence_threshold_ = std::clamp(threshold, 0.0f, 1.0f);
    impl_->model_info_.confidence_threshold = impl_->confidence_threshold_;
}

void InferenceEngine::SetNMSThreshold(float threshold) {
    impl_->nms_threshold_ = std::clamp(threshold, 0.0f, 1.0f);
    impl_->model_info_.nms_threshold = impl_->nms_threshold_;
}

std::string InferenceEngine::GetLastError() const {
    return impl_->last_error_;
}

std::vector<VeDeviceType> InferenceEngine::GetSupportedDevices() const {
    return {VE_DEVICE_CPU, VE_DEVICE_GPU, VE_DEVICE_CUDA};
}

std::vector<VePrecision> InferenceEngine::GetSupportedPrecisions(VeDeviceType device) const {
    return {VE_PRECISION_FP32, VE_PRECISION_FP16, VE_PRECISION_INT8};
}

// C API实现
VeInferenceHandle ve_inference_create(const VeEngineOptions* options) {
    if (!options) return nullptr;
    
    auto engine = new InferenceEngine();
    if (engine->Initialize(*options) != VE_SUCCESS) {
        delete engine;
        return nullptr;
    }
    
    return static_cast<VeInferenceHandle>(engine);
}

void ve_inference_destroy(VeInferenceHandle handle) {
    if (handle) {
        delete static_cast<InferenceEngine*>(handle);
    }
}

VeStatusCode ve_inference_load_model(VeInferenceHandle handle, const char* model_path) {
    if (!handle || !model_path) return VE_ERROR_INVALID_ARG;
    auto engine = static_cast<InferenceEngine*>(handle);
    return engine->LoadModel(model_path);
}

void ve_inference_unload_model(VeInferenceHandle handle) {
    if (handle) {
        auto engine = static_cast<InferenceEngine*>(handle);
        engine->UnloadModel();
    }
}

bool ve_inference_is_model_loaded(VeInferenceHandle handle) {
    if (!handle) return false;
    auto engine = static_cast<InferenceEngine*>(handle);
    return engine->IsModelLoaded();
}

VeInferenceResult* ve_inference_infer(VeInferenceHandle handle, const VeImageData* image) {
    if (!handle || !image) return nullptr;
    auto engine = static_cast<InferenceEngine*>(handle);
    auto result = engine->Infer(*image);
    
    auto* cr = new VeInferenceResult();
    cr->inference_time_ms = static_cast<float>(result->GetInferenceTimeMs());
    return cr;
}

void ve_inference_infer_async(VeInferenceHandle handle,
                               const VeImageData* image,
                               VeInferenceCallback callback) {
    if (!handle || !image || !callback) return;
    auto engine = static_cast<InferenceEngine*>(handle);
    std::async(std::launch::async, [engine, image, callback]() {
        auto result = engine->Infer(*image);
        VeInferenceResult* cr = new VeInferenceResult();
        cr->inference_time_ms = static_cast<float>(result->GetInferenceTimeMs());
        callback(cr);
    });
}

void ve_inference_set_callback(VeInferenceHandle handle, VeInferenceCallback callback) {
    if (handle) {
        auto engine = static_cast<InferenceEngine*>(handle);
        engine->SetResultCallback([callback](std::shared_ptr<InferenceResult> result) {
            VeInferenceResult* cr = new VeInferenceResult();
            cr->inference_time_ms = static_cast<float>(result->GetInferenceTimeMs());
            callback(cr);
        });
    }
}

VeInferenceResult** ve_inference_infer_batch(VeInferenceHandle handle,
                                             const VeImageData* images,
                                             int32_t batch_size,
                                             int32_t* result_count) {
    if (!handle || !images || batch_size <= 0) {
        if (result_count) *result_count = 0;
        return nullptr;
    }
    
    auto engine = static_cast<InferenceEngine*>(handle);
    auto batch_result = engine->InferBatch(images, batch_size);
    
    auto* results = new VeInferenceResult*[batch_size];
    *result_count = batch_size;
    
    for (int32_t i = 0; i < batch_size; ++i) {
        auto result = batch_result->GetResult(i);
        results[i] = new VeInferenceResult();
        results[i]->inference_time_ms = static_cast<float>(result->GetInferenceTimeMs());
    }
    
    return results;
}

VeStatusCode ve_inference_warmup(VeInferenceHandle handle) {
    if (!handle) return VE_ERROR_INVALID_ARG;
    auto engine = static_cast<InferenceEngine*>(handle);
    return engine->Warmup();
}

VePerformanceMetrics ve_inference_get_metrics(VeInferenceHandle handle) {
    if (!handle) return VePerformanceMetrics{};
    auto engine = static_cast<InferenceEngine*>(handle);
    return engine->GetPerformanceMetrics();
}

} // namespace vision_engine
