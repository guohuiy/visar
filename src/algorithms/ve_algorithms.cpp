/**
 * @brief VisionEngine 算法模块实现
 * @file ve_algorithms.cpp
 * @author VisionEngine Team
 * @date 2024
 */

#include "../../include/vision_engine/algorithms/ve_algorithms.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <numeric>
#include <iostream>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// 简单的日志宏
#define VE_LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define VE_LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl
#define VE_LOG_DEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl

namespace vision_engine {

// ============================================================================
// 工具函数
// ============================================================================

namespace utils {

bool FileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

float Clamp(float val, float min_val, float max_val) {
    return std::max(min_val, std::min(max_val, val));
}

float CalculateIoU(const VeBBox& a, const VeBBox& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

} // namespace utils

// ============================================================================
// YOLO类别名称
// ============================================================================

const std::map<int, std::string> kYOLOv8ClassNames = {
    {0, "person"}, {1, "bicycle"}, {2, "car"}, {3, "motorcycle"},
    {4, "airplane"}, {5, "bus"}, {6, "train"}, {7, "truck"},
    {8, "boat"}, {9, "traffic light"}, {10, "fire hydrant"},
    {11, "stop sign"}, {12, "parking meter"}, {13, "bench"},
    {14, "bird"}, {15, "cat"}, {16, "dog"}, {17, "horse"},
    {18, "sheep"}, {19, "cow"}, {20, "elephant"}, {21, "bear"},
    {22, "zebra"}, {23, "giraffe"}, {24, "backpack"}, {25, "umbrella"},
    {26, "handbag"}, {27, "tie"}, {28, "suitcase"}, {29, "frisbee"},
    {30, "skis"}, {31, "snowboard"}, {32, "sports ball"}, {33, "kite"},
    {34, "baseball bat"}, {35, "baseball glove"}, {36, "skateboard"},
    {37, "surfboard"}, {38, "tennis racket"}, {39, "bottle"},
    {40, "wine glass"}, {41, "cup"}, {42, "fork"}, {43, "knife"},
    {44, "spoon"}, {45, "bowl"}, {46, "banana"}, {47, "apple"},
    {48, "sandwich"}, {49, "orange"}, {50, "broccoli"}, {51, "carrot"},
    {52, "hot dog"}, {53, "pizza"}, {54, "donut"}, {55, "cake"},
    {56, "chair"}, {57, "couch"}, {58, "potted plant"}, {59, "bed"},
    {60, "dining table"}, {61, "toilet"}, {62, "tv"}, {63, "laptop"},
    {64, "mouse"}, {65, "remote"}, {66, "keyboard"}, {67, "cell phone"},
    {68, "microwave"}, {69, "oven"}, {70, "toaster"}, {71, "sink"},
    {72, "refrigerator"}, {73, "book"}, {74, "clock"}, {75, "vase"},
    {76, "scissors"}, {77, "teddy bear"}, {78, "hair drier"}, {79, "toothbrush"}
};

// ============================================================================
// NMS辅助函数
// ============================================================================

static std::vector<ObjectDetector::Detection> NMS(
    const std::vector<ObjectDetector::Detection>& detections,
    float iou_threshold) {
    
    std::vector<ObjectDetector::Detection> result;
    if (detections.empty()) return result;
    
    std::vector<ObjectDetector::Detection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(),
              [](const ObjectDetector::Detection& a, const ObjectDetector::Detection& b) {
                  return a.score > b.score;
              });
    
    std::vector<bool> suppressed(sorted.size(), false);
    
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (suppressed[j]) continue;
            float iou = utils::CalculateIoU(sorted[i].bbox, sorted[j].bbox);
            if (iou > iou_threshold) suppressed[j] = true;
        }
    }
    return result;
}

// ============================================================================
// ObjectDetector 基类实现
// ============================================================================

VeStatusCode ObjectDetector::Initialize(const std::string& model_path, const Config& config) {
    VE_LOG_INFO("Initializing ObjectDetector with model: " + model_path);
    if (!utils::FileExists(model_path)) {
        VE_LOG_ERROR("Model file not found: " + model_path);
        return VE_ERROR_FILE_NOT_FOUND;
    }
    config_ = config;
    VE_LOG_INFO("ObjectDetector initialized successfully");
    return VE_SUCCESS;
}

std::vector<ObjectDetector::Detection> ObjectDetector::Detect(const VeImageData& image) {
    return Detect(image.data, image.width, image.height, image.format);
}

void ObjectDetector::SetConfidenceThreshold(float threshold) {
    config_.confidence_threshold = utils::Clamp(threshold, 0.0f, 1.0f);
}

void ObjectDetector::SetNMSThreshold(float threshold) {
    config_.nms_threshold = utils::Clamp(threshold, 0.0f, 1.0f);
}

// ============================================================================
// YOLODetector 实现
// ============================================================================

YOLODetector::YOLODetector() {
    version_ = Version::YOLOv8;
    class_names_ = std::vector<std::string>(80);
    for (const auto& [id, name] : kYOLOv8ClassNames) {
        if (id < 80) class_names_[id] = name;
    }
}

VeStatusCode YOLODetector::Initialize(const std::string& model_path, const Config& config) {
    VE_LOG_INFO("Initializing YOLODetector (v8) with model: " + model_path);
    return ObjectDetector::Initialize(model_path, config);
}

void YOLODetector::SetModelVersion(Version version) {
    version_ = version;
}

std::vector<ObjectDetector::Detection> YOLODetector::Detect(const uint8_t* image_data,
                                                             int width, int height,
                                                             VeImageFormat format) {
    VE_LOG_DEBUG("YOLODetector::Detect called");
    std::vector<Detection> detections;
    
    if (!image_data || width <= 0 || height <= 0) {
        VE_LOG_ERROR("Invalid image data");
        return detections;
    }
    
    // 预处理
    const int target_size = 640;
    float scale_x = static_cast<float>(target_size) / width;
    float scale_y = static_cast<float>(target_size) / height;
    
    // 模拟推理输出 - 实际项目中应调用推理引擎
    std::vector<float> output(100 * 85, 0.0f);
    
    // 后处理
    const int num_classes = 80;
    const int stride = 4 + 1 + num_classes;
    const int num_boxes = static_cast<int>(output.size()) / stride;
    
    for (int i = 0; i < num_boxes; ++i) {
        const float* box_output = output.data() + i * stride;
        float cx = box_output[0];
        float cy = box_output[1];
        float w = box_output[2];
        float h = box_output[3];
        float obj_score = box_output[4];
        
        float max_class_score = 0.0f;
        int max_class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            if (box_output[5 + c] > max_class_score) {
                max_class_score = box_output[5 + c];
                max_class_id = c;
            }
        }
        
        float conf = obj_score * max_class_score;
        if (conf < config_.confidence_threshold) continue;
        
        float x1 = (cx - w * 0.5f) / scale_x;
        float y1 = (cy - h * 0.5f) / scale_y;
        float x2 = (cx + w * 0.5f) / scale_x;
        float y2 = (cy + h * 0.5f) / scale_y;
        
        Detection det;
        det.class_id = max_class_id;
        det.score = conf;
        det.bbox = {x1, y1, x2, y2};
        detections.push_back(det);
    }
    
    return NMS(detections, config_.nms_threshold);
}

// ============================================================================
// OCRRecognizer 实现
// ============================================================================

VeStatusCode OCRRecognizer::Initialize(const std::string& det_model_path,
                                       const std::string& rec_model_path,
                                       const Config& config) {
    VE_LOG_INFO("Initializing OCRRecognizer");
    config_ = config;
    VE_LOG_INFO("OCRRecognizer initialized successfully");
    return VE_SUCCESS;
}

std::vector<OCRRecognizer::TextDetection> OCRRecognizer::DetectText(const uint8_t* image_data,
                                                                     int width, int height) {
    VE_LOG_DEBUG("OCRRecognizer::DetectText called");
    return {};
}

std::vector<OCRRecognizer::TextRecognition> OCRRecognizer::Recognize(const uint8_t* image_data,
                                                                     int width, int height) {
    VE_LOG_DEBUG("OCRRecognizer::Recognize called");
    std::vector<TextRecognition> results;
    TextRecognition rec;
    rec.text = "Hello World";
    rec.score = 0.95f;
    rec.bbox = {0, 0, static_cast<float>(width), static_cast<float>(height)};
    results.push_back(rec);
    return results;
}

std::vector<OCRRecognizer::TextDetection> OCRRecognizer::DetectAndRecognize(const uint8_t* image_data,
                                                                           int width, int height) {
    return DetectText(image_data, width, height);
}

// ============================================================================
// AlgorithmPluginManager 实现
// ============================================================================

AlgorithmPluginManager& AlgorithmPluginManager::Instance() {
    static AlgorithmPluginManager instance;
    return instance;
}

std::unique_ptr<ObjectDetector> AlgorithmPluginManager::CreateDetector(const std::string& name) {
    auto it = detectors_.find(name);
    if (it != detectors_.end()) return it->second();
    return nullptr;
}

std::unique_ptr<ImageSegmenter> AlgorithmPluginManager::CreateSegmenter(const std::string& name) {
    auto it = segmenters_.find(name);
    if (it != segmenters_.end()) return it->second();
    return nullptr;
}

std::unique_ptr<OCRRecognizer> AlgorithmPluginManager::CreateOCR(const std::string& name) {
    auto it = ocrs_.find(name);
    if (it != ocrs_.end()) return it->second();
    return nullptr;
}

void AlgorithmPluginManager::ListAvailableAlgorithms() {
    VE_LOG_INFO("Available Object Detectors, Image Segmenters, and OCR Recognizers");
}

} // namespace vision_engine
