#include "ve_inference.h"
#include "ve_logger.h"

namespace vision_engine {

class InferenceEngine::Impl {
public:
    VeEngineOptions options_;
    bool model_loaded_ = false;
    std::string last_error_;
    VeModelInfo model_info_;
    ResultCallback result_callback_;
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
    impl_->options_.SetConfidenceThreshold(threshold);
}

void InferenceEngine::SetNMSThreshold(float threshold) {
    impl_->options_.SetNMSThreshold(threshold);
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
    EngineOptions opts;
    opts.SetPreferredBackend(options->preferred_backend);
    opts.SetDeviceType(options->device_type);
    opts.SetPrecision(options->precision);
    opts.SetNumThreads(options->num_threads);
    
    if (engine->Initialize(opts) != VE_SUCCESS) {
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
