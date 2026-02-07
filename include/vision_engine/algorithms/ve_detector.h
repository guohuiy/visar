#ifndef VE_DETECTOR_H
#define VE_DETECTOR_H

#include "../core/ve_types.h"
#include "ve_nms.h"
#include <vector>
#include <string>
#include <cmath>
#include <memory>

namespace vision_engine {
namespace postprocess {

// COCO数据集类别名称
inline const char* GetCOCOClassName(int class_id) {
    static const char* coco_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
        "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", 
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", 
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", 
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    
    if (class_id >= 0 && class_id < 80) {
        return coco_names[class_id];
    }
    return "unknown";
}

/**
 * @brief YOLO输出解析器
 */
class YOLOParser {
public:
    struct Config {
        float confidence_threshold = 0.5f;
        float nms_threshold = 0.45f;
        int num_classes = 80;
        bool use_ciou = false;
        int max_detections = 100;
    };
    
    YOLOParser() = default;
    explicit YOLOParser(const Config& config) : config_(config) {}
    
    void SetConfig(const Config& config) { config_ = config; }
    
    /**
     * @brief 解析YOLOv8/v5输出
     */
    std::vector<VeDetection> ParseYOLOv8(
        const float* output,
        const std::vector<int64_t>& output_shape,
        int original_width,
        int original_height,
        float scale_x = 1.0f,
        float scale_y = 1.0f,
        int pad_x = 0,
        int pad_y = 0) {
        
        std::vector<VeDetection> detections;
        
        if (output_shape.size() < 3) {
            return detections;
        }
        
        int num_anchors = 1;
        for (size_t i = 1; i < output_shape.size(); ++i) {
            num_anchors *= output_shape[i];
        }
        
        int num_params = config_.num_classes + 4 + 1;
        
        for (int i = 0; i < num_anchors; ++i) {
            const float* ptr = output + i * num_params;
            float obj_score = ptr[4];
            
            float max_class_score = 0;
            int class_id = -1;
            
            for (int c = 0; c < config_.num_classes; ++c) {
                float conf = ptr[5 + c] * obj_score;
                if (conf > max_class_score) {
                    max_class_score = conf;
                    class_id = c;
                }
            }
            
            if (max_class_score > config_.confidence_threshold) {
                float cx = ptr[0];
                float cy = ptr[1];
                float w = ptr[2];
                float h = ptr[3];
                
                float x1 = (cx - w * 0.5f) * scale_x - pad_x;
                float y1 = (cy - h * 0.5f) * scale_y - pad_y;
                float x2 = (cx + w * 0.5f) * scale_x - pad_x;
                float y2 = (cy + h * 0.5f) * scale_y - pad_y;
                
                x1 = std::max(0.0f, std::min(static_cast<float>(original_width) - 1.0f, x1));
                y1 = std::max(0.0f, std::min(static_cast<float>(original_height) - 1.0f, y1));
                x2 = std::max(0.0f, std::min(static_cast<float>(original_width) - 1.0f, x2));
                y2 = std::max(0.0f, std::min(static_cast<float>(original_height) - 1.0f, y2));
                
                VeDetection det;
                det.bbox = {x1, y1, x2, y2};
                det.score = max_class_score;
                det.class_id = class_id;
                det.class_name = GetCOCOClassName(class_id);
                det.keypoints = nullptr;
                det.num_keypoints = 0;
                det.mask = nullptr;
                
                detections.push_back(det);
            }
        }
        
        std::vector<VeDetection> result;
        if (config_.use_ciou) {
            std::sort(detections.begin(), detections.end(),
                [](const VeDetection& a, const VeDetection& b) {
                    return a.score > b.score;
                });
            std::vector<bool> keep(detections.size(), true);
            for (size_t i = 0; i < detections.size(); ++i) {
                if (!keep[i]) continue;
                result.push_back(detections[i]);
                for (size_t j = i + 1; j < detections.size(); ++j) {
                    if (!keep[j]) continue;
                    float iou = CalculateCIoU(result.back().bbox, detections[j].bbox);
                    if (iou > config_.nms_threshold) keep[j] = false;
                }
            }
        } else {
            result = NonMaximumSuppression(std::move(detections), config_.nms_threshold);
        }
        
        if (static_cast<int>(result.size()) > config_.max_detections) {
            result.resize(config_.max_detections);
        }
        
        return result;
    }
    
private:
    Config config_;
};

/**
 * @brief 检测结果可视化
 */
class DetectionVisualizer {
public:
    static std::vector<uint8_t> DrawDetections(
        std::vector<uint8_t> image_data,
        int width,
        int height,
        const std::vector<VeDetection>& detections) {
        
        static const uint8_t colors[8][3] = {
            {230, 126, 34}, {46, 204, 113}, {52, 152, 219}, {155, 89, 182},
            {244, 67, 54}, {0, 121, 107}, {255, 235, 59}, {0, 131, 210}
        };
        
        for (const auto& det : detections) {
            int color_idx = det.class_id % 8;
            const auto& color = colors[color_idx];
            
            // 绘制边界框
            int x1 = static_cast<int>(det.bbox.x1);
            int y1 = static_cast<int>(det.bbox.y1);
            int x2 = static_cast<int>(det.bbox.x2);
            int y2 = static_cast<int>(det.bbox.y2);
            
            x1 = std::max(0, std::min(width - 1, x1));
            y1 = std::max(0, std::min(height - 1, y1));
            x2 = std::max(0, std::min(width - 1, x2));
            y2 = std::max(0, std::min(height - 1, y2));
            
            int thickness = 2;
            
            // 上边框
            for (int x = x1; x <= x2 && x < width; ++x) {
                for (int dy = 0; dy < thickness && y1 + dy < height; ++dy) {
                    int idx = (y1 + dy) * width + x;
                    if (idx >= 0 && idx < static_cast<int>(image_data.size())) {
                        image_data[idx * 3] = color[0];
                        image_data[idx * 3 + 1] = color[1];
                        image_data[idx * 3 + 2] = color[2];
                    }
                }
            }
            
            // 下边框
            for (int x = x1; x <= x2 && x < width; ++x) {
                for (int dy = 0; dy < thickness && y2 - dy >= 0; ++dy) {
                    int idx = (y2 - dy) * width + x;
                    if (idx >= 0 && idx < static_cast<int>(image_data.size())) {
                        image_data[idx * 3] = color[0];
                        image_data[idx * 3 + 1] = color[1];
                        image_data[idx * 3 + 2] = color[2];
                    }
                }
            }
            
            // 左边框
            for (int y = y1; y <= y2 && y < height; ++y) {
                for (int dx = 0; dx < thickness && x1 + dx < width; ++dx) {
                    int idx = y * width + (x1 + dx);
                    if (idx >= 0 && idx < static_cast<int>(image_data.size())) {
                        image_data[idx * 3] = color[0];
                        image_data[idx * 3 + 1] = color[1];
                        image_data[idx * 3 + 2] = color[2];
                    }
                }
            }
            
            // 右边框
            for (int y = y1; y <= y2 && y < height; ++y) {
                for (int dx = 0; dx < thickness && x2 - dx >= 0; ++dx) {
                    int idx = y * width + (x2 - dx);
                    if (idx >= 0 && idx < static_cast<int>(image_data.size())) {
                        image_data[idx * 3] = color[0];
                        image_data[idx * 3 + 1] = color[1];
                        image_data[idx * 3 + 2] = color[2];
                    }
                }
            }
        }
        
        return image_data;
    }
};

} // namespace postprocess
} // namespace vision_engine

#endif // VE_DETECTOR_H
