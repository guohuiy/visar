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

// ========== 杈呭姪鍑芥暟锛氬浘鍍忛澶勭悊 ==========
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

// ========== 杈呭姪鍑芥暟锛歂MS鍚庡鐞?==========
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
