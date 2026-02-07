// Disable ONNX Runtime includes if not available
#ifdef HAVE_ONNX
#include "onnxruntime_cxx_api.h"
#endif

#include "ve_inference.h"
#include "ve_result.h"
#include "ve_model.h"
#include "../core/ve_logger.h"
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstring>

namespace vision_engine {

#ifdef HAVE_ONNX
// ONNX Runtime Session wrapper
class ONNXSession {
public:
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    bool Load(const std::string& model_path, Ort::SessionOptions& options) {
        try {
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
            session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), options);
            
            memory_info_ = std::make_unique<Ort::MemoryInfo>("Arena", OrtDeviceAllocator, 0, OrtMemTypeDefault);
            
            // Get input/output names and shapes
            size_t num_inputs = session_->GetInputCount();
            size_t num_outputs = session_->GetOutputCount();
            
            input_names_.resize(num_inputs);
            input_shapes_.resize(num_inputs);
            output_names_.resize(num_outputs);
            output_shapes_.resize(num_outputs);
            
            for (size_t i = 0; i < num_inputs; i++) {
                auto input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                input_names_[i] = strdup(input_name.get());
                Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
                auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                input_shapes_[i] = input_tensor_info.GetShape();
            }
            
            for (size_t i = 0; i < num_outputs; i++) {
                auto output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                output_names_[i] = strdup(output_name.get());
                Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
                auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
                output_shapes_[i] = output_tensor_info.GetShape();
            }
            
            return true;
        } catch (const Ort::Exception& e) {
            VE_LOG_ERROR(std::string("ONNX Exception: ") + e.what());
            return false;
        }
    }
    
    void Unload() {
        if (session_) {
            session_.reset();
        }
        for (auto name : input_names_) {
            if (name) free(const_cast<char*>(name));
        }
        for (auto name : output_names_) {
            if (name) free(const_cast<char*>(name));
        }
        input_names_.clear();
        output_names_.clear();
        input_shapes_.clear();
        output_shapes_.clear();
    }
    
    // Run inference
    std::vector<std::vector<float>> Run(
        const std::vector<const char*>& input_names,
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<const void*>& input_data,
        const std::vector<const char*>& output_names,
        const std::vector<std::vector<int64_t>>& output_shapes) {
        
        std::vector<std::vector<float>> outputs;
        
        try {
            std::vector<Ort::Value> ort_inputs;
            for (size_t i = 0; i < input_names.size(); i++) {
                size_t num_elements = 1;
                for (auto dim : input_shapes[i]) num_elements *= dim;
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
            
            for (size_t i = 0; i < output_tensors.size(); i++) {
                float* output_data = output_tensors[i].GetTensorMutableData<float>();
                auto& shape = output_shapes[i];
                size_t total_elements = 1;
                for (auto dim : shape) total_elements *= dim;
                outputs.push_back(std::vector<float>(output_data, output_data + total_elements));
            }
        } catch (const Ort::Exception& e) {
            VE_LOG_ERROR(std::string("ONNX Runtime Error: ") + e.what());
        }
        
        return outputs;
    }
};
#endif

// ========== 辅助函数：图像预处理 ==========
namespace preprocess {

// Bilinear interpolation resize
void ResizeImage(const uint8_t* src, int src_w, int src_h,
                 float* dst, int dst_w, int dst_h, int channels) {
    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;
    
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            float src_x = (x + 0.5f) * scale_x - 0.5f;
            float src_y = (y + 0.5f) * scale_y - 0.5f;
            
            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, src_w - 1);
            int y1 = std::min(y0 + 1, src_h - 1);
            
            float fx = src_x - x0;
            float fy = src_y - y0;
            
            for (int c = 0; c < channels; c++) {
                float v00 = static_cast<float>(src[(y0 * src_w + x0) * channels + c]);
                float v01 = static_cast<float>(src[(y0 * src_w + x1) * channels + c]);
                float v10 = static_cast<float>(src[(y1 * src_w + x0) * channels + c]);
                float v11 = static_cast<float>(src[(y1 * src_w + x1) * channels + c]);
                
                float val = v00 * (1 - fx) * (1 - fy) +
                           v01 * fx * (1 - fy) +
                           v10 * (1 - fx) * fy +
                           v11 * fx * fy;
                
                dst[(y * dst_w + x) * channels + c] = val;
            }
        }
    }
}

// Normalize image data (default: ImageNet normalization)
void Normalize(float* data, size_t size, const float* mean, const float* std, bool has_mean_std) {
    const float* mean_ptr = has_mean_std ? mean : nullptr;
    const float* std_ptr = has_mean_std ? std : nullptr;
    
    if (!has_mean_std) {
        // Default ImageNet normalization
        static const float default_mean[3] = {123.675f, 116.28f, 103.53f};
        static const float default_std[3] = {58.395f, 57.12f, 57.375f};
        mean_ptr = default_mean;
        std_ptr = default_std;
    }
    
    for (size_t i = 0; i < size; i += 3) {
        data[i] = (data[i] - mean_ptr[0]) / std_ptr[0];
        data[i + 1] = (data[i + 1] - mean_ptr[1]) / std_ptr[1];
        data[i + 2] = (data[i + 2] - mean_ptr[2]) / std_ptr[2];
    }
}

// Convert RGB/BGR to tensor format (HWC -> CHW)
void HWCtoCHW(float* src, float* dst, int height, int width, int channels) {
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                dst[c * height * width + h * width + w] = 
                    src[h * width * channels + w * channels + c];
            }
        }
    }
}

} // namespace preprocess

// ========== 辅助函数：NMS后处理 ==========
namespace postprocess {

float IoU(const VeBBox& a, const VeBBox& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

std::vector<VeDetection> NonMaximumSuppression(
    std::vector<VeDetection> detections,
    float iou_threshold) {
    
    std::vector<VeDetection> result;
    
    std::sort(detections.begin(), detections.end(),
        [](const VeDetection& a, const VeDetection& b) {
            return a.score > b.score;
        });
    
    while (!detections.empty()) {
        result.push_back(detections[0]);
        if (detections.size() == 1) break;
        
        std::vector<VeDetection> remaining;
        for (size_t i = 1; i < detections.size(); i++) {
            if (IoU(result.back().bbox, detections[i].bbox) < iou_threshold) {
                remaining.push_back(detections[i]);
            }
        }
        detections = std::move(remaining);
    }
    
    return result;
}

std::vector<VeDetection> ProcessYOLOv8Output(
    const float* output, const std::vector<int64_t>& output_shape,
    int input_w, int input_h,
    float confidence_threshold,
    int num_classes) {
    
    std::vector<VeDetection> detections;
    
    int num_anchors = 1;
    for (size_t i = 1; i < output_shape.size(); i++) {
        num_anchors *= output_shape[i];
    }
    
    int stride = num_classes + 4;
    
    for (int i = 0; i < num_anchors; i++) {
        const float* ptr = output + i * stride;
        float obj_score = ptr[4];
        
        float max_class_score = 0;
        int class_id = -1;
        for (int c = 0; c < num_classes; c++) {
            float score = ptr[5 + c] * obj_score;
            if (score > max_class_score) {
                max_class_score = score;
                class_id = c;
            }
        }
        
        if (max_class_score > confidence_threshold) {
            float cx = ptr[0];
            float cy = ptr[1];
            float w = ptr[2];
            float h = ptr[3];
            
            VeBBox bbox;
            bbox.x1 = cx - w * 0.5f;
            bbox.y1 = cy - h * 0.5f;
            bbox.x2 = cx + w * 0.5f;
            bbox.y2 = cy + h * 0.5f;
            
            bbox.x1 = std::max(0.0f, std::min(static_cast<float>(input_w) - 1.0f, bbox.x1));
            bbox.y1 = std::max(0.0f, std::min(static_cast<float>(input_h) - 1.0f, bbox.y1));
            bbox.x2 = std::max(0.0f, std::min(static_cast<float>(input_w) - 1.0f, bbox.x2));
            bbox.y2 = std::max(0.0f, std::min(static_cast<float>(input_h) - 1.0f, bbox.y2));
            
            VeDetection det;
            det.bbox = bbox;
            det.score = max_class_score;
            det.class_id = class_id;
            det.class_name = nullptr;
            det.keypoints = nullptr;
            det.num_keypoints = 0;
            det.mask = nullptr;
            detections.push_back(det);
        }
    }
    
    return detections;
}

} // namespace postprocess

class InferenceEngine::Impl {
public:
    VeEngineOptions options_;
    bool model_loaded_ = false;
    std::string last_error_;
    VeModelInfo model_info_;
    ResultCallback result_callback_;
    float confidence_threshold_ = 0.5f;
    float nms_threshold_ = 0.45f;
    
#ifdef HAVE_ONNX
    std::unique_ptr<ONNXSession> onnx_session_;
#endif
    
    int input_width_ = 640;
    int input_height_ = 640;
    int input_channels_ = 3;
};

InferenceEngine::InferenceEngine() : impl_(std::make_unique<Impl>()) {}

InferenceEngine::~InferenceEngine() {
#ifdef HAVE_ONNX
    impl_->onnx_session_.reset();
#endif
    UnloadModel();
}

VeStatusCode InferenceEngine::Initialize(const VeEngineOptions& options) {
    impl_->options_ = options;
    VE_LOG_INFO("InferenceEngine initialized");
    return VE_SUCCESS;
}

VeStatusCode InferenceEngine::LoadModel(const std::string& model_path) {
    VE_LOG_INFO("Loading model: " + model_path);
    
#ifdef HAVE_ONNX
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        impl_->last_error_ = "Model file not found: " + model_path;
        return VE_ERROR_FILE_NOT_FOUND;
    }
    file.close();
    
    try {
        impl_->onnx_session_ = std::make_unique<ONNXSession>();
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(impl_->options_.num_threads > 0 ? impl_->options_.num_threads : 4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (!impl_->onnx_session_->Load(model_path, session_options)) {
            impl_->last_error_ = "Failed to load ONNX model";
            impl_->onnx_session_.reset();
            return VE_ERROR_MODEL_LOAD_FAILED;
        }
        
        if (!impl_->onnx_session_->input_shapes_.empty()) {
            auto& input_shape = impl_->onnx_session_->input_shapes_[0];
            if (input_shape.size() >= 4) {
                impl_->input_channels_ = static_cast<int>(input_shape[1]);
                impl_->input_height_ = static_cast<int>(input_shape[2]);
                impl_->input_width_ = static_cast<int>(input_shape[3]);
            }
        }
        
        impl_->model_loaded_ = true;
        impl_->model_info_.path = model_path.c_str();
        impl_->model_info_.name = "ONNX Model";
        impl_->model_info_.backend = VE_BACKEND_ONNX;
        impl_->model_info_.input_width = impl_->input_width_;
        impl_->model_info_.input_height = impl_->input_height_;
        impl_->model_info_.input_channels = impl_->input_channels_;
        
        VE_LOG_INFO("Model loaded successfully");
        return VE_SUCCESS;
    } catch (const std::exception& e) {
        impl_->last_error_ = std::string("Model load error: ") + e.what();
        return VE_ERROR_MODEL_LOAD_FAILED;
    }
#else
    // Without ONNX, just mark as loaded for demo
    impl_->model_loaded_ = true;
    impl_->model_info_.path = model_path.c_str();
    impl_->model_info_.name = "Demo Model";
    VE_LOG_WARN("ONNX Runtime not available, using demo mode");
    return VE_SUCCESS;
#endif
}

void InferenceEngine::UnloadModel() {
#ifdef HAVE_ONNX
    impl_->onnx_session_.reset();
#endif
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
    
    impl_->model_info_.input_width = image.width;
    impl_->model_info_.input_height = image.height;
    
#ifdef HAVE_ONNX
    if (!impl_->onnx_session_ || !impl_->onnx_session_->session_) {
        VE_LOG_WARN("ONNX session not available, returning empty result");
        return result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 1. Preprocess: Resize image to model input size
    int target_w = impl_->input_width_;
    int target_h = impl_->input_height_;
    int channels = impl_->input_channels_;
    size_t tensor_size = static_cast<size_t>(target_w) * target_h * channels;
    
    std::vector<float> resized_data(tensor_size);
    preprocess::ResizeImage(image.data, image.width, image.height,
                           resized_data.data(), target_w, target_h, channels);
    
    // 2. Normalize
    preprocess::Normalize(resized_data.data(), tensor_size, image.mean, image.std, 
                          image.mean != nullptr && image.std != nullptr);
    
    // 3. HWC -> CHW
    std::vector<float> chw_data(tensor_size);
    preprocess::HWCtoCHW(resized_data.data(), chw_data.data(), target_h, target_w, channels);
    
    // 4. Run ONNX inference
    auto outputs = impl_->onnx_session_->Run(
        impl_->onnx_session_->input_names_,
        impl_->onnx_session_->input_shapes_,
        {(const void*)chw_data.data()},
        impl_->onnx_session_->output_names_,
        impl_->onnx_session_->output_shapes_
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // 设置推理时间
    result->SetInferenceTimeMs(inference_time);
    
    // 5. Postprocess: Parse detections
    if (!outputs.empty() && !outputs[0].empty()) {
        auto detections = postprocess::ProcessYOLOv8Output(
            outputs[0].data(),
            impl_->onnx_session_->output_shapes_[0],
            target_w,
            target_h,
            impl_->confidence_threshold_,
            80  // Default 80 classes (COCO)
        );
        
        // 6. Apply NMS
        detections = postprocess::NonMaximumSuppression(std::move(detections), impl_->nms_threshold_);
        
        // 7. Set results
        for (const auto& det : detections) {
            result->AddDetection(det);
        }
    }
#endif
    
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
    auto future = std::async(std::launch::async, [engine, image, callback]() {
        auto result = engine->Infer(*image);
        VeInferenceResult* cr = new VeInferenceResult();
        cr->inference_time_ms = static_cast<float>(result->GetInferenceTimeMs());
        callback(cr);
    });
    (void)future;  // Suppress unused variable warning
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
