/**
 * @file ve_fast_nms.h
 * @brief 快速NMS算法 - 使用空间索引加速
 * @author VisionEngine Team
 * @date 2024-02
 */

#ifndef VE_FAST_NMS_H
#define VE_FAST_NMS_H

#include "../core/ve_types.h"
#include "ve_nms.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <queue>

namespace vision_engine {
namespace postprocess {

/**
 * @brief 快速NMS配置
 */
struct FastNMSConfig {
    float iou_threshold = 0.45f;
    int max_detections = 100;
    int grid_size = 64;          // 空间网格大小
    bool use_ciou = false;       // 是否使用CIoU
};

/**
 * @brief 使用空间哈希索引的快速NMS
 * 
 * 复杂度: O(n) 平均情况, 而不是 O(n²)
 * 适用于高密度检测场景
 */
class FastNMS {
public:
    FastNMS() : config_(FastNMSConfig{}) {}
    
    explicit FastNMS(const FastNMSConfig& config) : config_(config) {}
    
    void SetConfig(const FastNMSConfig& config) {
        config_ = config;
    }
    
    /**
     * @brief 执行快速NMS
     * @param detections 输入检测结果
     * @return NMS过滤后的结果
     */
    std::vector<VeDetection> Process(
        std::vector<VeDetection> detections) {
        
        if (detections.empty()) {
            return {};
        }
        
        // 1. 按置信度排序
        std::sort(detections.begin(), detections.end(),
            [](const VeDetection& a, const VeDetection& b) {
                return a.score > b.score;
            });
        
        // 2. 建立空间网格索引
        SpatialHashGrid grid(config_.grid_size);
        std::vector<VeDetection*> indexed_dets;
        indexed_dets.reserve(detections.size());
        
        for (auto& det : detections) {
            indexed_dets.push_back(&det);
            int gx1 = static_cast<int>(det.bbox.x1) / config_.grid_size;
            int gy1 = static_cast<int>(det.bbox.y1) / config_.grid_size;
            int gx2 = static_cast<int>(det.bbox.x2) / config_.grid_size;
            int gy2 = static_cast<int>(det.bbox.y2) / config_.grid_size;
            
            // 扩大搜索范围 (+1 网格)
            for (int gx = std::max(0, gx1 - 1); gx <= gx2 + 1; ++gx) {
                for (int gy = std::max(0, gy1 - 1); gy <= gy2 + 1; ++gy) {
                    int key = gy * 100000 + gx;  // 唯一键
                    grid.Insert(key, &det);
                }
            }
        }
        
        // 3. 快速NMS
        std::vector<bool> suppressed(detections.size(), false);
        std::vector<VeDetection> result;
        result.reserve(config_.max_detections);
        
        for (size_t i = 0; i < indexed_dets.size() && 
             static_cast<int>(result.size()) < config_.max_detections; ++i) {
            
            if (suppressed[i]) continue;
            
            VeDetection* current = indexed_dets[i];
            result.push_back(*current);
            
            // 查找相邻网格中的检测
            int gx1 = static_cast<int>(current->bbox.x1) / config_.grid_size;
            int gy1 = static_cast<int>(current->bbox.y1) / config_.grid_size;
            int gx2 = static_cast<int>(current->bbox.x2) / config_.grid_size;
            int gy2 = static_cast<int>(current->bbox.y2) / config_.grid_size;
            
            for (int gx = std::max(0, gx1 - 1); gx <= gx2 + 1; ++gx) {
                for (int gy = std::max(0, gy1 - 1); gy <= gy2 + 1; ++gy) {
                    int key = gy * 100000 + gx;
                    auto nearby = grid.Get(key);
                    
                    for (VeDetection* other : nearby) {
                        if (other == current) continue;
                        
                        // 找到对应的索引
                        size_t other_idx = other - &detections[0];
                        if (other_idx <= i || suppressed[other_idx]) continue;
                        
                        // 计算IoU
                        float iou;
                        if (config_.use_ciou) {
                            iou = CalculateCIoU(current->bbox, other->bbox);
                        } else {
                            iou = CalculateIoU(current->bbox, other->bbox);
                        }
                        
                        if (iou > config_.iou_threshold) {
                            suppressed[other_idx] = true;
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
private:
    /**
     * @brief 空间哈希网格
     */
    class SpatialHashGrid {
    public:
        explicit SpatialHashGrid(int cell_size) : cell_size_(cell_size) {}
        
        void Insert(int key, VeDetection* det) {
            grid_[key].push_back(det);
        }
        
        std::vector<VeDetection*> Get(int key) {
            auto it = grid_.find(key);
            if (it != grid_.end()) {
                return it->second;
            }
            return {};
        }
        
        void Clear() {
            grid_.clear();
        }
        
    private:
        int cell_size_;
        std::unordered_map<int, std::vector<VeDetection*>> grid_;
    };
    
    FastNMSConfig config_;
};

/**
 * @brief 批量NMS - 合并多个检测结果后执行NMS
 */
inline std::vector<VeDetection> BatchNMS(
    std::vector<std::vector<VeDetection>>& detections_list,
    float iou_threshold = 0.45f,
    int max_detections = 100) {
    
    std::vector<VeDetection> merged;
    
    // 合并所有检测
    for (auto& dets : detections_list) {
        merged.insert(merged.end(), dets.begin(), dets.end());
    }
    
    // 执行NMS
    FastNMSConfig config;
    config.iou_threshold = iou_threshold;
    config.max_detections = max_detections;
    
    FastNMS nms(config);
    return nms.Process(std::move(merged));
}

/**
 * @brief 带置信度加权的NMS
 * 
 * 对低置信度检测使用更严格的IoU阈值
 */
class WeightedNMS {
public:
    WeightedNMS() : config_(FastNMSConfig{}) {}
    
    explicit WeightedNMS(const FastNMSConfig& config) : config_(config) {}
    
    std::vector<VeDetection> Process(
        std::vector<VeDetection> detections) {
        
        if (detections.empty()) {
            return {};
        }
        
        // 按置信度排序
        std::sort(detections.begin(), detections.end(),
            [](const VeDetection& a, const VeDetection& b) {
                return a.score > b.score;
            });
        
        std::vector<bool> suppressed(detections.size(), false);
        std::vector<VeDetection> result;
        
        for (size_t i = 0; i < detections.size(); ++i) {
            if (suppressed[i]) continue;
            
            const VeDetection& current = detections[i];
            result.push_back(current);
            
            // 根据当前检测的置信度动态调整阈值
            float adaptive_threshold = config_.iou_threshold;
            if (current.score < 0.3f) {
                adaptive_threshold = config_.iou_threshold * 0.7f;  // 更严格
            }
            
            for (size_t j = i + 1; j < detections.size(); ++j) {
                if (suppressed[j]) continue;
                
                float iou = CalculateIoU(current.bbox, detections[j].bbox);
                if (iou > adaptive_threshold) {
                    suppressed[j] = true;
                }
            }
            
            if (static_cast<int>(result.size()) >= config_.max_detections) {
                break;
            }
        }
        
        return result;
    }
    
private:
    FastNMSConfig config_;
};

/**
 * @brief 检测结果融合器
 * 
 * 将多个模型的检测结果进行融合
 */
class DetectionFuser {
public:
    struct FuseConfig {
        float iou_threshold = 0.5f;
        float score_threshold = 0.3f;
        int max_detections = 100;
        bool weighted_average = true;
    };
    
    explicit DetectionFuser(const FuseConfig& config) : config_(config) {}
    
    /**
     * @brief 融合多个检测结果
     * @param detections_list 多个模型的检测结果
     * @return 融合后的检测结果
     */
    std::vector<VeDetection> Fuse(
        const std::vector<std::vector<VeDetection>>& detections_list) {
        
        if (detections_list.empty()) {
            return {};
        }
        
        // 合并所有检测
        std::vector<VeDetection> all_detections;
        for (const auto& dets : detections_list) {
            for (const auto& det : dets) {
                if (det.score >= config_.score_threshold) {
                    all_detections.push_back(det);
                }
            }
        }
        
        if (all_detections.empty()) {
            return {};
        }
        
        // 按置信度排序
        std::sort(all_detections.begin(), all_detections.end(),
            [](const VeDetection& a, const VeDetection& b) {
                return a.score > b.score;
            });
        
        // 分组和融合
        std::vector<bool> fused(all_detections.size(), false);
        std::vector<VeDetection> result;
        
        for (size_t i = 0; i < all_detections.size(); ++i) {
            if (fused[i]) continue;
            
            std::vector<const VeDetection*> group;
            group.push_back(&all_detections[i]);
            fused[i] = true;
            
            // 找到所有重叠的检测
            for (size_t j = i + 1; j < all_detections.size(); ++j) {
                if (fused[j]) continue;
                
                float iou = CalculateIoU(all_detections[i].bbox, all_detections[j].bbox);
                if (iou > config_.iou_threshold) {
                    group.push_back(&all_detections[j]);
                    fused[j] = true;
                }
            }
            
            // 融合该组
            if (group.size() > 1) {
                VeDetection fused_det = FuseGroup(group);
                result.push_back(fused_det);
            } else {
                result.push_back(all_detections[i]);
            }
            
            if (static_cast<int>(result.size()) >= config_.max_detections) {
                break;
            }
        }
        
        return result;
    }
    
private:
    FuseConfig config_;
    
    /**
     * @brief 融合一组检测
     */
    VeDetection FuseGroup(const std::vector<const VeDetection*>& group) {
        VeDetection result = *group[0];
        
        if (config_.weighted_average) {
            // 加权平均边界框
            float total_score = 0;
            float avg_x1 = 0, avg_y1 = 0, avg_x2 = 0, avg_y2 = 0;
            
            for (const auto* det : group) {
                total_score += det->score;
                avg_x1 += det->bbox.x1 * det->score;
                avg_y1 += det->bbox.y1 * det->score;
                avg_x2 += det->bbox.x2 * det->score;
                avg_y2 += det->bbox.y2 * det->score;
            }
            
            result.bbox.x1 = avg_x1 / total_score;
            result.bbox.y1 = avg_y1 / total_score;
            result.bbox.x2 = avg_x2 / total_score;
            result.bbox.y2 = avg_y2 / total_score;
            
            // 平均置信度 (加权)
            result.score = total_score / group.size();
        }
        
        return result;
    }
};

} // namespace postprocess
} // namespace vision_engine

#endif // VE_FAST_NMS_H
