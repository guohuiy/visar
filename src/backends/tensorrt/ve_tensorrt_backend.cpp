#include "ve_tensorrt_backend.h"
#include "../../core/ve_logger.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>

#ifdef HAVE_TENSORRT
#include <NvInferRuntime.h>
#include <NvOnnxParserRuntime.h>
#endif

namespace vision_engine {

#ifdef HAVE_TENSORRT

// TensorRT Logger
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINFO:
                VE_LOG_INFO(std::string("[TensorRT] ") + msg);
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                VE_LOG_WARN(std::string("[TensorRT] ") + msg);
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                VE_LOG_ERROR(std::string("[TensorRT] ") + msg);
                break;
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            case nvinfer1::ILogger::Severity::kERROR:
                VE_LOG_ERROR(std::string("[TensorRT] ") + msg);
                break;
            default:
                break;
        }
    }
};

// TensorRT Buffer Manager
class TensorRTBufferManager {
public:
    TensorRTBufferManager(nvinfer1::ICudaEngine& engine, int max_batch_size = 1) {
        // 分配输入/输出缓冲区
        for (int i = 0; i < engine.getNbBindings(); ++i) {
            auto binding_name = engine.getBindingName(i);
            auto binding_dims = engine.getBindingDimensions(i);
            
            // 计算缓冲区大小
            size_t vol = 1;
            for (int j = 0; j < binding_dims.nbDims; ++j) {
                vol *= binding_dims.d[j];
            }
            vol *= max_batch_size;
            
            // 确定是否为输入
            bool is_input = engine.bindingIsInput(i);
            
            // 分配CPU和GPU内存
            void* cpu_buffer = malloc(vol * sizeof(float));
            void* gpu_buffer = nullptr;
            
            cudaMalloc(&gpu_buffer, vol * sizeof(float));
            cudaMemset(gpu_buffer, 0, vol * sizeof(float));
            
            if (is_input) {
                cpu_buffers_.push_back(cpu_buffer);
                gpu_buffers_.push_back(gpu_buffer);
            } else {
                cpu_buffers_.push_back(cpu_buffer);
                gpu_buffers_.push_back(gpu_buffer);
            }
        }
    }
    
    ~TensorRTBufferManager() {
        for (auto ptr : cpu_buffers_) free(ptr);
        for (auto ptr : gpu_buffers_) {
            cudaFree(ptr);
        }
    }
    
    void* GetCPUBuffer(int index) { return cpu_buffers_[index]; }
    void* GetGPUBuffer(int index) { return gpu_buffers_[index]; }
    size_t GetBufferSize(int index) const { 
        // 返回缓冲区大小
        return 0; 
    }
    
private:
    std::vector<void*> cpu_buffers_;
    std::vector<void*> gpu_buffers_;
};

class TensorRTBackend::Impl {
public:
    std::unique_ptr<TensorRTLogger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<TensorRTBufferManager> buffer_manager_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    bool fp16_enabled_ = false;
    bool int8_enabled_ = false;
    size_t workspace_size_ = 4096; // MB
    int max_batch_size_ = 1;
    
    bool LoadEngine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.is_open()) {
            VE_LOG_ERROR("Cannot open TensorRT engine file: " + engine_path);
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();
        
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(*logger_));
        
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(engine_data.data(), size));
        
        if (!engine_) {
            VE_LOG_ERROR("Failed to deserialize TensorRT engine");
            return false;
        }
        
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        
        // 初始化缓冲区
        buffer_manager_ = std::make_unique<TensorRTBufferManager>(*engine_, max_batch_size_);
        
        // 获取输入输出名称和形状
        input_names_.clear();
        output_names_.clear();
        input_shapes_.clear();
        output_shapes_.clear();
        
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            auto binding_dims = engine_->getBindingDimensions(i);
            std::vector<int64_t> shape(binding_dims.d, binding_dims.d + binding_dims.nbDims);
            
            if (engine_->bindingIsInput(i)) {
                input_names_.push_back(engine_->getBindingName(i));
                input_shapes_.push_back(shape);
            } else {
                output_names_.push_back(engine_->getBindingName(i));
                output_shapes_.push_back(shape);
            }
        }
        
        return true;
    }
    
    bool BuildEngine(const std::string& onnx_path) {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(*logger_));
        
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(1 << static_cast<int>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        
        auto parser = std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, *logger_));
        
        if (!parser->parseFromFile(onnx_path.c_str(), 
                                   static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            VE_LOG_ERROR("Failed to parse ONNX model");
            return false;
        }
        
        // 构建配置
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
            builder->createBuilderConfig());
        
        config->setMaxWorkspaceSize(workspace_size_ * 1024 * 1024);
        
        if (fp16_enabled_) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        
        if (int8_enabled_) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
        }
        
        config->setMaxBatchSize(max_batch_size_);
        
        // 构建引擎
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config));
        
        if (!engine_) {
            VE_LOG_ERROR("Failed to build TensorRT engine");
            return false;
        }
        
        // 创建运行时和上下文
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(*logger_));
        
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        
        // 初始化缓冲区
        buffer_manager_ = std::make_unique<TensorRTBufferManager>(*engine_, max_batch_size_);
        
        return true;
    }
};

#endif // HAVE_TENSORRT

TensorRTBackend::TensorRTBackend() {
#ifdef HAVE_TENSORRT
    impl_ = std::make_unique<Impl>();
    impl_->logger_ = std::make_unique<TensorRTLogger>();
#else
    VE_LOG_WARN("TensorRT is not available, using stub implementation");
#endif
}

TensorRTBackend::~TensorRTBackend() = default;

VeStatusCode TensorRTBackend::Initialize(const std::string& model_path, 
                                         const VeEngineOptions& options) {
#ifdef HAVE_TENSORRT
    try {
        impl_->max_batch_size_ = options.batch_size > 0 ? options.batch_size : 1;
        
        // 检查文件扩展名
        std::string extension = model_path.substr(model_path.find_last_of(".") + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        bool success = false;
        if (extension == "trt" || extension == "engine") {
            success = impl_->LoadEngine(model_path);
        } else if (extension == "onnx") {
            success = impl_->BuildEngine(model_path);
        } else {
            VE_LOG_ERROR("Unsupported model format: " + extension);
            return VE_ERROR_INVALID_ARG;
        }
        
        if (success) {
            VE_LOG_INFO("TensorRT backend initialized successfully");
            return VE_SUCCESS;
        }
        
        return VE_ERROR_MODEL_LOAD_FAILED;
    } catch (const std::exception& e) {
        VE_LOG_ERROR(std::string("TensorRT initialization error: ") + e.what());
        return VE_ERROR_MODEL_LOAD_FAILED;
    }
#else
    VE_LOG_WARN("TensorRT not available");
    return VE_ERROR_BACKEND_NOT_SUPPORTED;
#endif
}

VeStatusCode TensorRTBackend::Infer(const float* input_data,
                                    const std::vector<int64_t>& input_shape,
                                    float* output_data) {
#ifdef HAVE_TENSORRT
    if (!impl_->context_ || !impl_->buffer_manager_) {
        return VE_ERROR_INVALID_STATE;
    }
    
    try {
        // 复制输入数据到GPU
        void* gpu_input = impl_->buffer_manager_->GetGPUBuffer(0);
        cudaMemcpy(gpu_input, input_data, 
                   input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * sizeof(float),
                   cudaMemcpyHostToDevice);
        
        // 执行推理
        bool success = impl_->context_->executeV2(&impl_->buffer_manager_->GetGPUBuffer(0));
        
        if (!success) {
            VE_LOG_ERROR("TensorRT inference failed");
            return VE_ERROR_INFERENCE_FAILED;
        }
        
        // 复制输出数据回CPU
        void* gpu_output = impl_->buffer_manager_->GetGPUBuffer(1);
        cudaMemcpy(output_data, gpu_output,
                   impl_->output_shapes_[0][0] * impl_->output_shapes_[0][1] * 
                   impl_->output_shapes_[0][2] * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        return VE_SUCCESS;
    } catch (const std::exception& e) {
        VE_LOG_ERROR(std::string("TensorRT inference error: ") + e.what());
        return VE_ERROR_INFERENCE_FAILED;
    }
#else
    return VE_ERROR_BACKEND_NOT_SUPPORTED;
#endif
}

VeStatusCode TensorRTBackend::InferINT8(const uint8_t* input_data,
                                       const std::vector<int64_t>& input_shape,
                                       float* output_data) {
#ifdef HAVE_TENSORRT
    if (!impl_->int8_enabled_) {
        VE_LOG_WARN("INT8 not enabled, falling back to FP32");
        return Infer(reinterpret_cast<const float*>(input_data), input_shape, output_data);
    }
    return Infer(reinterpret_cast<const float*>(input_data), input_shape, output_data);
#else
    return VE_ERROR_BACKEND_NOT_SUPPORTED;
#endif
}

std::vector<std::string> TensorRTBackend::GetInputNames() const {
#ifdef HAVE_TENSORRT
    return impl_->input_names_;
#else
    return {};
#endif
}

std::vector<std::string> TensorRTBackend::GetOutputNames() const {
#ifdef HAVE_TENSORRT
    return impl_->output_names_;
#else
    return {};
#endif
}

std::vector<int64_t> TensorRTBackend::GetInputShape() const {
#ifdef HAVE_TENSORRT
    if (!impl_->input_shapes_.empty()) {
        return impl_->input_shapes_[0];
    }
#endif
    return {};
}

std::vector<int64_t> TensorRTBackend::GetOutputShape() const {
#ifdef HAVE_TENSORRT
    if (!impl_->output_shapes_.empty()) {
        return impl_->output_shapes_[0];
    }
#endif
    return {};
}

void TensorRTBackend::EnableFP16() {
#ifdef HAVE_TENSORRT
    impl_->fp16_enabled_ = true;
#endif
}

void TensorRTBackend::EnableINT8(const std::vector<std::vector<float>>& calib_data) {
#ifdef HAVE_TENSORRT
    impl_->int8_enabled_ = true;
    // 校准数据可以在此处理
    (void)calib_data;
#endif
}

void TensorRTBackend::SetWorkspaceSize(size_t size_mb) {
#ifdef HAVE_TENSORRT
    impl_->workspace_size_ = size_mb;
#endif
}

size_t TensorRTBackend::GetDeviceMemoryUsage() const {
#ifdef HAVE_TENSORRT
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    return total_memory - free_memory;
#else
    return 0;
#endif
}

void TensorRTBackend::Warmup(int iterations) {
#ifdef HAVE_TENSORRT
    VE_LOG_INFO("Warming up TensorRT engine...");
    for (int i = 0; i < iterations; ++i) {
        // 创建随机输入进行预热
        auto input_shape = GetInputShape();
        if (!input_shape.empty()) {
            std::vector<float> dummy_input(1 * 3 * 640 * 640, 0.0f);
            std::vector<float> dummy_output(1 * 100 * 6, 0.0f);
            Infer(dummy_input.data(), input_shape, dummy_output.data());
        }
    }
#endif
}

std::string TensorRTBackend::GetEngineInfo() const {
#ifdef HAVE_TENSORRT
    std::ostringstream oss;
    oss << "TensorRT Engine Info:\n";
    oss << "  Inputs: " << impl_->input_names_.size() << "\n";
    for (size_t i = 0; i < impl_->input_names_.size(); ++i) {
        oss << "    " << impl_->input_names_[i] << ": ";
        for (auto dim : impl_->input_shapes_[i]) oss << dim << " ";
        oss << "\n";
    }
    oss << "  Outputs: " << impl_->output_names_.size() << "\n";
    for (size_t i = 0; i < impl_->output_names_.size(); ++i) {
        oss << "    " << impl_->output_names_[i] << ": ";
        for (auto dim : impl_->output_shapes_[i]) oss << dim << " ";
        oss << "\n";
    }
    oss << "  FP16: " << (impl_->fp16_enabled_ ? "enabled" : "disabled") << "\n";
    oss << "  INT8: " << (impl_->int8_enabled_ ? "enabled" : "disabled") << "\n";
    return oss.str();
#else
    return "TensorRT not available";
#endif
}

} // namespace vision_engine
