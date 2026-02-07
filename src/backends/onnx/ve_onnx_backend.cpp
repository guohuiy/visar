#include "ve_onnx_backend.h"
#include "../../core/ve_logger.h"
#include "../../core/ve_error.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <numeric>

#ifdef HAVE_ONNX
#include "onnxruntime_cxx_api.h"
#endif

namespace vision_engine {

#ifdef HAVE_ONNX

class ONNXBackend::Impl {
public:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs_;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs_;
    
    bool cuda_enabled_ = false;
    int num_threads_ = 4;
    
    bool LoadModel(const std::string& model_path) {
        try {
            // 创建环境
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VisionEngine-ONNX");
            
            // 创建会话选项
            session_options_ = std::make_unique<Ort::SessionOptions>();
            session_options_->SetIntraOpNumThreads(num_threads_);
            session_options_->SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // 启用CUDA（如果可用）
            if (cuda_enabled_) {
                try {
                    OrtCUDAProviderOptions cuda_options;
                    session_options_->AppendExecutionProvider_CUDA(cuda_options);
                    VE_LOG_INFO("CUDA execution provider enabled");
                } catch (const Ort::Exception& e) {
                    VE_LOG_WARN(std::string("CUDA not available: ") + e.what());
                }
            }
            
            // 创建会话
            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
            
            // 创建内存信息
            memory_info_ = std::make_unique<Ort::MemoryInfo>("Arena", OrtDeviceAllocator, 0, OrtMemTypeDefault);
            
            // 获取输入输出信息
            size_t num_inputs = session_->GetInputCount();
            size_t num_outputs = session_->GetOutputCount();
            
            input_names_.resize(num_inputs);
            input_shapes_.resize(num_inputs);
            input_name_ptrs_.resize(num_inputs);
            
            output_names_.resize(num_outputs);
            output_shapes_.resize(num_outputs);
            output_name_ptrs_.resize(num_outputs);
            
            for (size_t i = 0; i < num_inputs; ++i) {
                auto name_ptr = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                input_name_ptrs_[i] = std::move(name_ptr);
                input_names_[i] = input_name_ptrs_[i].get();
                
                auto type_info = session_->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                input_shapes_[i] = tensor_info.GetShape();
            }
            
            for (size_t i = 0; i < num_outputs; ++i) {
                auto name_ptr = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                output_name_ptrs_[i] = std::move(name_ptr);
                output_names_[i] = output_name_ptrs_[i].get();
                
                auto type_info = session_->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                output_shapes_[i] = tensor_info.GetShape();
            }
            
            return true;
        } catch (const Ort::Exception& e) {
            VE_LOG_ERROR(std::string("ONNX Runtime Exception: ") + e.what());
            return false;
        }
    }
    
    std::vector<std::vector<float>> RunInference(
        const std::vector<const char*>& input_names,
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<const void*>& input_data,
        const std::vector<const char*>& output_names,
        const std::vector<std::vector<int64_t>>& output_shapes) {
        
        std::vector<std::vector<float>> outputs;
        
        try {
            std::vector<Ort::Value> ort_inputs;
            for (size_t i = 0; i < input_names.size(); ++i) {
                size_t num_elements = 1;
                for (auto dim : input_shapes[i]) {
                    if (dim > 0) num_elements *= dim;
                }
                
                ort_inputs.push_back(Ort::Value::CreateTensor(
                    *memory_info_,
                    const_cast<void*>(input_data[i]),
                    num_elements * sizeof(float),
                    input_shapes[i].data(),
                    input_shapes[i].size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                ));
            }
            
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names.data(),
                ort_inputs.data(),
                ort_inputs.size(),
                output_names.data(),
                output_names.size()
            );
            
            outputs.resize(output_tensors.size());
            for (size_t i = 0; i < output_tensors.size(); ++i) {
                float* data = output_tensors[i].GetTensorMutableData<float>();
                auto& shape = output_shapes[i];
                size_t num_elements = 1;
                for (auto dim : shape) {
                    if (dim > 0) num_elements *= dim;
                }
                outputs[i].assign(data, data + num_elements);
            }
        } catch (const Ort::Exception& e) {
            VE_LOG_ERROR(std::string("ONNX Inference Error: ") + e.what());
        }
        
        return outputs;
    }
};

#endif // HAVE_ONNX

ONNXBackend::ONNXBackend() {
#ifdef HAVE_ONNX
    impl_ = std::make_unique<Impl>();
#else
    VE_LOG_WARN("ONNX Runtime is not available, using stub implementation");
#endif
}

ONNXBackend::~ONNXBackend() = default;

VeStatusCode ONNXBackend::Initialize(const std::string& model_path) {
#ifdef HAVE_ONNX
    try {
        if (!impl_->LoadModel(model_path)) {
            return VE_ERROR_MODEL_LOAD_FAILED;
        }
        VE_LOG_INFO("ONNX backend initialized successfully");
        return VE_SUCCESS;
    } catch (const std::exception& e) {
        VE_LOG_ERROR(std::string("ONNX initialization error: ") + e.what());
        return VE_ERROR_MODEL_LOAD_FAILED;
    }
#else
    return VE_ERROR_BACKEND_NOT_SUPPORTED;
#endif
}

VeStatusCode ONNXBackend::Infer(const float* input_data,
                               const std::vector<int64_t>& input_shape,
                               float* output_data) {
#ifdef HAVE_ONNX
    if (!impl_->session_) {
        return VE_ERROR_INVALID_STATE;
    }
    
    try {
        std::vector<const char*> input_names_vec(impl_->input_names_.size());
        std::vector<const char*> output_names_vec(impl_->output_names_.size());
        std::vector<const void*> input_data_vec;
        std::vector<std::vector<int64_t>> input_shapes_vec;
        
        for (size_t i = 0; i < impl_->input_names_.size(); ++i) {
            input_names_vec[i] = impl_->input_names_[i].c_str();
            input_data_vec.push_back(input_data);
            input_shapes_vec.push_back(input_shape);
        }
        
        for (size_t i = 0; i < impl_->output_names_.size(); ++i) {
            output_names_vec[i] = impl_->output_names_[i].c_str();
        }
        
        auto outputs = impl_->RunInference(
            input_names_vec,
            input_shapes_vec,
            input_data_vec,
            output_names_vec,
            impl_->output_shapes_
        );
        
        if (!outputs.empty() && output_data) {
            size_t output_size = std::accumulate(
                outputs[0].begin(), outputs[0].end(), 0ULL,
                [](size_t sum, float v) { return sum + 1; }
            );
            std::copy(outputs[0].begin(), outputs[0].end(), output_data);
        }
        
        return VE_SUCCESS;
    } catch (const std::exception& e) {
        VE_LOG_ERROR(std::string("ONNX inference error: ") + e.what());
        return VE_ERROR_INFERENCE_FAILED;
    }
#else
    return VE_ERROR_BACKEND_NOT_SUPPORTED;
#endif
}

std::vector<std::string> ONNXBackend::GetInputNames() const {
#ifdef HAVE_ONNX
    return impl_->input_names_;
#else
    return {};
#endif
}

std::vector<std::string> ONNXBackend::GetOutputNames() const {
#ifdef HAVE_ONNX
    return impl_->output_names_;
#else
    return {};
#endif
}

std::vector<int64_t> ONNXBackend::GetInputShape() const {
#ifdef HAVE_ONNX
    if (!impl_->input_shapes_.empty()) {
        return impl_->input_shapes_[0];
    }
#endif
    return {};
}

std::vector<int64_t> ONNXBackend::GetOutputShape() const {
#ifdef HAVE_ONNX
    if (!impl_->output_shapes_.empty()) {
        return impl_->output_shapes_[0];
    }
#endif
    return {};
}

void ONNXBackend::SetExecutionProviders(const std::vector<std::string>& providers) {
#ifdef HAVE_ONNX
    (void)providers;
    // 可以在此实现多个执行提供商的设置
#endif
}

void ONNXBackend::EnableCUDA() {
#ifdef HAVE_ONNX
    impl_->cuda_enabled_ = true;
#endif
}

} // namespace vision_engine
