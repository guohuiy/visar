#ifndef VE_NMS_H
#define VE_NMS_H

#include "../core/ve_types.h"
#include <vector>
#include <algorithm>

#ifdef __cplusplus
#include <cmath>
#include <functional>

namespace vision_engine {
namespace postprocess {

/**
 * @brief NMS (Non-Maximum Suppression) 后处理模块
 * 提供多种NMS算法实现
 */

// IoU (Intersection over Union) 计算
inline float CalculateIoU(const VeBBox& a, const VeBBox& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * 
                       std::max(0.0f, inter_y2 - inter_y1);
    
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

// GIoU (Generalized IoU) 计算
inline float CalculateGIoU(const VeBBox& a, const VeBBox& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * 
                       std::max(0.0f, inter_y2 - inter_y1);
    
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    float iou = inter_area / (area_a + area_b - inter_area + 1e-6f);
    
    float convex_x1 = std::min(a.x1, b.x1);
    float convex_y1 = std::min(a.y1, b.y1);
    float convex_x2 = std::max(a.x2, b.x2);
    float convex_y2 = std::max(a.y2, b.y2);
    
    float convex_area = (convex_x2 - convex_x1) * (convex_y2 - convex_y1);
    
    float convex_union = convex_area - (area_a + area_b - inter_area);
    convex_union = std::max(convex_union, 1e-6f);
    
    return iou - (convex_union - (area_a + area_b - inter_area)) / convex_union;
}

// DIoU (Distance IoU) 计算
inline float CalculateDIoU(const VeBBox& a, const VeBBox& b) {
    float iou = CalculateIoU(a, b);
    
    float center_x1 = (a.x1 + a.x2) * 0.5f;
    float center_y1 = (a.y1 + a.y2) * 0.5f;
    float center_x2 = (b.x1 + b.x2) * 0.5f;
    float center_y2 = (b.y1 + b.y2) * 0.5f;
    
    float center_dist_sq = (center_x1 - center_x2) * (center_x1 - center_x2) +
                          (center_y1 - center_y2) * (center_y1 - center_y2);
    
    float convex_x1 = std::min(a.x1, b.x1);
    float convex_y1 = std::min(a.y1, b.y1);
    float convex_x2 = std::max(a.x2, b.x2);
    float convex_y2 = std::max(a.y2, b.y2);
    
    float convex_diag_sq = (convex_x2 - convex_x1) * (convex_x2 - convex_x1) +
                          (convex_y2 - convex_y1) * (convex_y2 - convex_y1);
    convex_diag_sq = std::max(convex_diag_sq, 1e-6f);
    
    return iou - center_dist_sq / convex_diag_sq;
}

// CIoU (Complete IoU) 计算
inline float CalculateCIoU(const VeBBox& a, const VeBBox& b) {
    float iou = CalculateIoU(a, b);
    float diou = CalculateDIoU(a, b);
    
    float w1 = a.x2 - a.x1;
    float h1 = a.y2 - a.y1;
    float w2 = b.x2 - b.x1;
    float h2 = b.y2 - b.y1;
    
    float ar = (4.0f / (3.14159f * 3.14159f)) * 
               std::atan(w1 / (h1 + 1e-6f)) - std::atan(w2 / (h2 + 1e-6f));
    ar = ar * ar;
    
    float v = 4.0f * ar * ar / ((3.14159f * 3.14159f) * (3.14159f * 3.14159f) + ar);
    float alpha = v / ((1 - iou) + v + 1e-6f);
    
    return diou - alpha * v;
}

/**
 * @brief 标准NMS实现
 * @param detections 检测结果数组
 * @param iou_threshold IoU阈值
 * @return 过滤后的检测结果
 */
inline std::vector<VeDetection> NonMaximumSuppression(
    std::vector<VeDetection> detections,
    float iou_threshold = 0.45f) {
    
    std::vector<VeDetection> result;
    
    if (detections.empty()) {
        return result;
    }
    
    // 按置信度排序
    std::sort(detections.begin(), detections.end(),
        [](const VeDetection& a, const VeDetection& b) {
            return a.score > b.score;
        });
    
    while (!detections.empty()) {
        // 选择置信度最高的
        result.push_back(detections[0]);
        
        if (detections.size() == 1) {
            break;
        }
        
        // 计算与其他检测的IoU
        std::vector<VeDetection> remaining;
        for (size_t i = 1; i < detections.size(); ++i) {
            float iou = CalculateIoU(result.back().bbox, detections[i].bbox);
            if (iou < iou_threshold) {
                remaining.push_back(detections[i]);
            }
        }
        detections = std::move(remaining);
    }
    
    return result;
}

/**
 * @brief 软NMS实现 (Soft-NMS)
 * @param detections 检测结果数组
 * @param iou_threshold IoU阈值
 * @param sigma Sigma参数 (用于高斯加权)
 * @param method 方法: 0-线性, 1-高斯
 * @return 过滤后的检测结果
 */
inline std::vector<VeDetection> SoftNMS(
    std::vector<VeDetection> detections,
    float iou_threshold = 0.45f,
    float sigma = 0.5f,
    int method = 1) { // 0: linear, 1: gaussian
    
    if (detections.empty()) {
        return detections;
    }
    
    // 按置信度排序
    std::sort(detections.begin(), detections.end(),
        [](const VeDetection& a, const VeDetection& b) {
            return a.score > b.score;
        });
    
    std::vector<float> scores(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        scores[i] = detections[i].score;
    }
    
    while (true) {
        // 找到最高分数的索引
        int max_idx = -1;
        float max_score = 0;
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            if (scores[i] > max_score) {
                max_score = scores[i];
                max_idx = i;
            }
        }
        
        if (max_idx < 0 || max_score < 0.001f) {
            break;
        }
        
        // 添加到结果
        detections[max_idx].score = max_score;
        
        // 抑制其他检测
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            if (i == max_idx) continue;
            
            float iou = CalculateIoU(detections[max_idx].bbox, detections[i].bbox);
            
            if (method == 0) { // 线性衰减
                if (iou > iou_threshold) {
                    scores[i] *= (1.0f - iou);
                }
            } else { // 高斯衰减
                scores[i] *= std::exp(-iou * iou / sigma);
            }
        }
        
        scores[max_idx] = 0;
    }
    
    // 移除低置信度检测
    std::vector<VeDetection> result;
    for (const auto& det : detections) {
        if (det.score > 0.001f) {
            result.push_back(det);
        }
    }
    
    return result;
}

/**
 * @brief 批量NMS (用于多尺度检测)
 * @param all_detections 所有检测结果
 * @param iou_threshold IoU阈值
 * @param max_detections 最大检测数量
 * @return 合并后的检测结果
 */
inline std::vector<VeDetection> BatchNMS(
    const std::vector<std::vector<VeDetection>>& all_detections,
    float iou_threshold = 0.45f,
    int max_detections = 100) {
    
    std::vector<VeDetection> merged;
    
    // 合并所有检测
    for (const auto& dets : all_detections) {
        merged.insert(merged.end(), dets.begin(), dets.end());
    }
    
    // 执行NMS
    merged = NonMaximumSuppression(std::move(merged), iou_threshold);
    
    // 限制数量
    if (static_cast<int>(merged.size()) > max_detections) {
        merged.resize(max_detections);
    }
    
    return merged;
}

/**
 * @brief 置信度阈值过滤
 * @param detections 检测结果
 * @param threshold 置信度阈值
 * @return 过滤后的检测结果
 */
inline std::vector<VeDetection> FilterByConfidence(
    std::vector<VeDetection> detections,
    float threshold = 0.5f) {
    
    std::vector<VeDetection> result;
    for (auto& det : detections) {
        if (det.score >= threshold) {
            result.push_back(det);
        }
    }
    return result;
}

/**
 * @brief 按类别NMS
 * @param detections 检测结果
 * @param iou_threshold IoU阈值
 * @return 按类别分组后的检测结果
 */
inline std::vector<VeDetection> ClasswiseNMS(
    std::vector<VeDetection> detections,
    float iou_threshold = 0.45f) {
    
    // 按类别分组
    std::unordered_map<int32_t, std::vector<VeDetection>> class_groups;
    for (const auto& det : detections) {
        class_groups[det.class_id].push_back(det);
    }
    
    std::vector<VeDetection> result;
    
    // 对每个类别执行NMS
    for (auto& group : class_groups) {
        auto filtered = NonMaximumSuppression(std::move(group.second), iou_threshold);
        result.insert(result.end(), filtered.begin(), filtered.end());
    }
    
    return result;
}

} // namespace postprocess
} // namespace vision_engine

#endif // __cplusplus

#endif // VE_NMS_H
