#include "ve_onnx_backend.h"

namespace vision_engine {

class ONNXBackend::Impl {
public:
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

ONNXBackend::ONNXBackend() : env_(ORT_LOGGING_LEVEL_WARNING, "VisionEngine-ONNX"), 
                              impl_(std::make_unique<Impl>()) {}

ONNXBackend::~ONNXBackend() = default;

VeStatusCode ONNXBackend::Initialize(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        impl_->session_ = std::make_unique<Ort::Session>(
            impl_->env_, model_path.c_str(), session_options);
        
        return VE_SUCCESS;
    } catch (const Ort::Exception& e) {
        return VE_ERROR_MODEL_LOAD_FAILED;
    }
}

std::vector<std::string> ONNXBackend::GetInputNames() const {
    return impl_->input_names_;
}

std::vector<std::string> ONNXBackend::GetOutputNames() const {
    return impl_->output_names_;
}

std::vector<int64_t> ONNXBackend::GetInputShape() const {
    return {};
}

std::vector<int64_t> ONNXBackend::GetOutputShape() const {
    return {};
}

} // namespace vision_engine
