/**
 * @file ve_multi_scale.h
 * @brief 多尺度特征融合模块
 * @author VisionEngine Team
 * @date 2024-02
 */

#ifndef VE_MULTI_SCALE_H
#define VE_MULTI_SCALE_H

#include "../core/ve_types.h"
#include "ve_nms.h"
#include "ve_detector.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace vision_engine {
namespace postprocess {

/**
 * @brief 多尺度特征融合配置
 */
struct MultiScaleConfig {
    // 锚框 strides (对应P3, P4, P5)
    std::vector<int> strides = {8, 16, 32};
    
    // 检测参数
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.45f;
    int num_classes = 80;
    bool use_ciou = true;
    int max_detections = 300;
    
    // 图像尺寸
    int input_width = 640;
    int input_height = 640;
    
    // 原始图像尺寸（用于坐标映射）
    int original_width = 640;
    int original_height = 640;
    
    // 预处理参数（Letterbox）
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

/**
 * @brief 多尺度特征融合检测器
 * 
 * 特性：
 * - 支持多尺度特征图融合 (P3/P4/P5)
 * - 自适应锚框匹配
 * - 跨尺度NMS
 */
class MultiScaleDetector {
public:
    MultiScaleDetector() = default;
    
    explicit MultiScaleDetector(const MultiScaleConfig& config) 
        : config_(config) {}
    
    void SetConfig(const MultiScaleConfig& config) {
        config_ = config;
    }
    
    /**
     * @brief 解析YOLOv8多尺度输出
     * @param outputs 各尺度输出 tensor 数组
     * @param shapes 各尺度输出 shape 数组
     * @return 融合后的检测结果
     */
    std::vector<VeDetection> Detect(
        const std::vector<std::vector<float>>& outputs,
        const std::vector<std::vector<int64_t>>& shapes) {
        
        std::vector<VeDetection> all_detections;
        
        // 1. 解析各个尺度的检测结果
        for (size_t i = 0; i < outputs.size() && i < config_.strides.size(); ++i) {
            auto dets = ParseSingleScale(
                outputs[i], 
                shapes[i],
                config_.strides[i]
            );
            all_detections.insert(all_detections.end(), dets.begin(), dets.end());
        }
        
        // 2. 坐标映射到原图尺寸
        MapToOriginalSize(all_detections);
        
        // 3. 按置信度排序
        std::sort(all_detections.begin(), all_detections.end(),
            [](const VeDetection& a, const VeDetection& b) {
                return a.score > b.score;
            });
        
        // 4. 跨尺度NMS
        std::vector<VeDetection> result;
        if (config_.use_ciou) {
            result = ClasswiseNMS(std::move(all_detections), config_.nms_threshold);
        } else {
            result = NonMaximumSuppression(std::move(all_detections), config_.nms_threshold);
        }
        
        // 5. 限制最大检测数量
        if (static_cast<int>(result.size()) > config_.max_detections) {
            result.resize(config_.max_detections);
        }
        
        return result;
    }
    
    /**
     * @brief 解析单尺度YOLO输出
     */
    std::vector<VeDetection> ParseSingleScale(
        const std::vector<float>& output,
        const std::vector<int64_t>& shape,
        int stride) {
        
        std::vector<VeDetection> detections;
        
        if (shape.size() < 4) {
            return detections;
        }
        
        int batch = static_cast<int>(shape[0]);
        int channels = static_cast<int>(shape[1]);
        int height = static_cast<int>(shape[2]);
        int width = static_cast<int>(shape[3]);
        
        int num_params = config_.num_classes + 4 + 1;  // class + box + obj
        int grid_size = height * width;
        
        // 计算该尺度对应的锚框尺寸
        float anchor_base = stride * 4.0f;  // 基础锚框大小
        
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < channels; ++c) {
                    int param_idx = c;
                    int class_offset = 5;
                    
                    float obj_score = output[param_idx + 4 * grid_size];
                    if (obj_score < config_.confidence_threshold) {
                        continue;
                    }
                    
                    // 找到最大类别分数
                    float max_class_score = 0;
                    int class_id = -1;
                    
                    for (int cls = 0; cls < config_.num_classes; ++cls) {
                        float score = output[param_idx + (class_offset + cls) * grid_size] * obj_score;
                        if (score > max_class_score) {
                            max_class_score = score;
                            class_id = cls;
                        }
                    }
                    
                    if (max_class_score > config_.confidence_threshold) {
                        // 解码边界框
                        float tx = output[param_idx];
                        float ty = output[param_idx + grid_size];
                        float tw = output[param_idx + 2 * grid_size];
                        float th = output[param_idx + 3 * grid_size];
                        
                        // 转换为图像坐标
                        float cx = (w + Sigmoid(tx)) * stride;
                        float cy = (h + Sigmoid(ty)) * stride;
                        
                        // 锚框宽高
                        float anchor_w = anchor_base * 0.5f;
                        float anchor_h = anchor_base * 0.5f;
                        
                        float bw = std::exp(tw) * anchor_w;
                        float bh = std::exp(th) * anchor_h;
                        
                        VeBBox bbox;
                        bbox.x1 = cx - bw * 0.5f;
                        bbox.y1 = cy - bh * 0.5f;
                        bbox.x2 = cx + bw * 0.5f;
                        bbox.y2 = cy + bh * 0.5f;
                        
                        // 裁剪到图像范围
                        bbox.x1 = std::max(0.0f, std::min(bbox.x1, static_cast<float>(config_.input_width)));
                        bbox.y1 = std::max(0.0f, std::min(bbox.y1, static_cast<float>(config_.input_height)));
                        bbox.x2 = std::max(0.0f, std::min(bbox.x2, static_cast<float>(config_.input_width)));
                        bbox.y2 = std::max(0.0f, std::min(bbox.y2, static_cast<float>(config_.input_height)));
                        
                        VeDetection det;
                        det.bbox = bbox;
                        det.score = max_class_score;
                        det.class_id = class_id;
                        det.class_name = GetCOCOClassName(class_id);
                        det.keypoints = nullptr;
                        det.num_keypoints = 0;
                        det.mask = nullptr;
                        
                        detections.push_back(det);
                    }
                }
            }
        }
        
        return detections;
    }
    
private:
    static float Sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    /**
     * @brief 将检测坐标映射回原图尺寸
     */
    void MapToOriginalSize(std::vector<VeDetection>& detections) {
        float scale_x = config_.scale;
        float scale_y = config_.scale;
        int pad_x = config_.pad_x;
        int pad_y = config_.pad_y;
        
        int orig_w = config_.original_width;
        int orig_h = config_.original_height;
        
        for (auto& det : detections) {
            // 反变换坐标
            float x1 = (det.bbox.x1 + pad_x) / scale_x;
            float y1 = (det.bbox.y1 + pad_y) / scale_y;
            float x2 = (det.bbox.x2 + pad_x) / scale_x;
            float y2 = (det.bbox.y2 + pad_y) / scale_y;
            
            // 裁剪到原图范围
            x1 = std::max(0.0f, std::min(static_cast<float>(orig_w) - 1.0f, x1));
            y1 = std::max(0.0f, std::min(static_cast<float>(orig_h) - 1.0f, y1));
            x2 = std::max(0.0f, std::min(static_cast<float>(orig_w) - 1.0f, x2));
            y2 = std::max(0.0f, std::min(static_cast<float>(orig_h) - 1.0f, y2));
            
            det.bbox = {x1, y1, x2, y2};
        }
    }
    
    MultiScaleConfig config_;
};

/**
 * @brief 小目标检测增强器
 * 
 * 特性：
 * - 高分辨率特征图检测小目标
 * - 多尺度测试
 */
class SmallObjectEnhancer {
public:
    struct EnhanceConfig {
        int roi_size = 128;          // ROI区域大小
        float min_object_size = 10;   // 小目标阈值
        float confidence_boost = 0.1f; // 小目标置信度提升
    };
    
    explicit SmallObjectEnhancer(const EnhanceConfig& config) 
        : config_(config) {}
    
    /**
     * @brief 增强小目标检测
     * @param original_detections 原始检测结果
     * @param image 原图数据
     * @param width 图像宽度
     * @param height 图像高度
     * @return 增强后的检测结果
     */
    std::vector<VeDetection> Enhance(
        std::vector<VeDetection> original_detections,
        const uint8_t* image,
        int width,
        int height) {
        
        // 1. 找出小目标区域
        std::vector<VeBBox> small_rois;
        for (const auto& det : original_detections) {
            float obj_w = det.bbox.x2 - det.bbox.x1;
            float obj_h = det.bbox.y2 - det.bbox.y1;
            
            if (obj_w < config_.min_object_size || obj_h < config_.min_object_size) {
                // 扩大ROI区域
                float expand = config_.roi_size * 0.5f;
                VeBBox roi;
                roi.x1 = std::max(0.0f, det.bbox.x1 - expand);
                roi.y1 = std::max(0.0f, det.bbox.y1 - expand);
                roi.x2 = std::min(static_cast<float>(width), det.bbox.x2 + expand);
                roi.y2 = std::min(static_cast<float>(height), det.bbox.y2 + expand);
                small_rois.push_back(roi);
            }
        }
        
        // 2. 对每个小目标区域执行超分辨率/增强推理
        for (const auto& roi : small_rois) {
            // TODO: 在此执行小目标增强推理
            // 可选：将ROI放大后重新检测
            
            // 临时：对小目标置信度进行加权提升
            for (auto& det : original_detections) {
                    if (CalculateIoU(det.bbox, roi) > 0.5f) {
                    det.score = std::min(1.0f, det.score + config_.confidence_boost);
                }
            }
        }
        
        return original_detections;
    }
    
private:
    EnhanceConfig config_;
};

} // namespace postprocess
} // namespace vision_engine

#endif // VE_MULTI_SCALE_H
