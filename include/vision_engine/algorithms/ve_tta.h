/**
 * @file ve_tta.h
 * @brief 测试时增强 (Test-Time Augmentation) 模块
 * @author VisionEngine Team
 * @date 2024-02
 */

#ifndef VE_TTA_H
#define VE_TTA_H

#include "../core/ve_types.h"
#include "ve_preprocess.h"
#include "ve_nms.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>

namespace vision_engine {
namespace tta {

// 内联IoU计算函数
inline float InlineIoU(const VeBBox& a, const VeBBox& b) {
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

/**
 * @brief TTA配置
 */
struct TTAConfig {
    // 启用的增强类型
    bool horizontal_flip = true;
    bool multi_scale = true;
    std::vector<float> scales = {0.83f, 1.0f, 1.17f};  // 对应640输入
    
    // 多尺度参数
    int base_size = 640;
    int max_size = 640;
    
    // 融合参数
    float score_threshold = 0.3f;
    float iou_threshold = 0.5f;
    int max_detections = 100;
    
    // 加权融合权重
    float original_weight = 1.0f;
    float flip_weight = 0.5f;
    float scale_weight = 0.33f;
};

/**
 * @brief 水平翻转图像
 */
inline std::vector<uint8_t> FlipHorizontal(
    const uint8_t* src, int width, int height, int channels = 3) {
    
    std::vector<uint8_t> dst(width * height * channels);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                dst[(y * width + x) * channels + c] = 
                    src[(y * width + (width - 1 - x)) * channels + c];
            }
        }
    }
    
    return dst;
}

/**
 * @brief 投票融合检测结果
 */
std::vector<VeDetection> VoteFusion(
    std::vector<std::vector<VeDetection>>& detections_list,
    const TTAConfig& config) {
    
    if (detections_list.empty()) {
        return {};
    }
    
    // 合并所有检测
    std::vector<VeDetection> all_detections;
    for (auto& dets : detections_list) {
        for (auto& det : dets) {
            if (det.score >= config.score_threshold) {
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
    
    // 加权融合
    struct VotedDetection {
        VeBBox bbox;
        float total_score = 0;
        int count = 0;
        int class_id = -1;
    };
    
    std::vector<VotedDetection> voted;
    std::vector<bool> processed(all_detections.size(), false);
    
    for (size_t i = 0; i < all_detections.size(); ++i) {
        if (processed[i]) continue;
        
        const VeDetection& det = all_detections[i];
        VotedDetection vd;
        vd.bbox = det.bbox;
        vd.total_score = det.score;
        vd.count = 1;
        vd.class_id = det.class_id;
        
        for (size_t j = i + 1; j < all_detections.size(); ++j) {
            if (processed[j]) continue;
            
            float iou = InlineIoU(det.bbox, all_detections[j].bbox);
            if (iou > config.iou_threshold && 
                all_detections[j].class_id == det.class_id) {
                
                processed[j] = true;
                vd.total_score += all_detections[j].score;
                vd.count++;
                
                // 加权平均边界框
                float w = all_detections[j].score;
                vd.bbox.x1 = vd.bbox.x1 * (1 - 1.0f/vd.count) + all_detections[j].bbox.x1 / vd.count;
                vd.bbox.y1 = vd.bbox.y1 * (1 - 1.0f/vd.count) + all_detections[j].bbox.y1 / vd.count;
                vd.bbox.x2 = vd.bbox.x2 * (1 - 1.0f/vd.count) + all_detections[j].bbox.x2 / vd.count;
                vd.bbox.y2 = vd.bbox.y2 * (1 - 1.0f/vd.count) + all_detections[j].bbox.y2 / vd.count;
            }
        }
        
        processed[i] = true;
        vd.total_score /= vd.count;  // 平均置信度
        voted.push_back(vd);
        
        if (static_cast<int>(voted.size()) >= config.max_detections) {
            break;
        }
    }
    
    // 转换为检测结果
    std::vector<VeDetection> result;
    for (const auto& vd : voted) {
        VeDetection det;
        det.bbox = vd.bbox;
        det.score = vd.total_score;
        det.class_id = vd.class_id;
        det.class_name = nullptr;
        det.keypoints = nullptr;
        det.num_keypoints = 0;
        det.mask = nullptr;
        result.push_back(det);
    }
    
    return result;
}

/**
 * @brief 测试时增强引擎
 */
class TTAEngine {
public:
    TTAEngine() : config_(TTAConfig{}) {}
    
    explicit TTAEngine(const TTAConfig& config) : config_(config) {}
    
    void SetConfig(const TTAConfig& config) {
        config_ = config;
    }
    
    /**
     * @brief 执行TTA推理
     * @param image 输入图像
     * @param width 图像宽度
     * @param height 图像高度
     * @param infer_func 推理函数回调
     * @return 融合后的检测结果
     */
    template<typename InferFunc>
    std::vector<VeDetection> Process(
        const uint8_t* image, int width, int height,
        InferFunc infer_func) {
        
        std::vector<std::vector<VeDetection>> all_results;
        preprocess::PreprocessConfig pp_config;
        
        // 1. 原始图像推理
        auto result_original = infer_func(image, width, height, 0, 0, 0);
        if (!result_original.empty()) {
            all_results.push_back(result_original);
        }
        
        // 2. 水平翻转推理
        if (config_.horizontal_flip) {
            auto flipped = FlipHorizontal(image, width, height);
            auto result_flipped = infer_func(flipped.data(), width, height, 
                                              1.0f, 1.0f, 0);
            // 翻转检测框坐标
            for (auto& det : result_flipped) {
                float temp = width - 1 - det.bbox.x1;
                det.bbox.x1 = width - 1 - det.bbox.x2;
                det.bbox.x2 = temp;
            }
            if (!result_flipped.empty()) {
                all_results.push_back(result_flipped);
            }
        }
        
        // 3. 多尺度推理
        if (config_.multi_scale) {
            for (float scale : config_.scales) {
                if (scale == 1.0f) continue;  // 跳过原始尺度
                
                int new_w = static_cast<int>(width * scale);
                int new_h = static_cast<int>(height * scale);
                
                // 缩放图像
                std::vector<uint8_t> scaled(new_w * new_h * 3);
                ScaleImage(image, width, height, scaled.data(), new_w, new_h);
                
                auto result_scaled = infer_func(scaled.data(), new_w, new_h, 
                                                 scale, scale, 0);
                
                // 映射回原始坐标
                for (auto& det : result_scaled) {
                    det.bbox.x1 /= scale;
                    det.bbox.y1 /= scale;
                    det.bbox.x2 /= scale;
                    det.bbox.y2 /= scale;
                }
                
                if (!result_scaled.empty()) {
                    all_results.push_back(result_scaled);
                }
            }
        }
        
        // 4. 投票融合
        return VoteFusion(all_results, config_);
    }
    
private:
    TTAConfig config_;
    
    /**
     * @brief 简单图像缩放 (最近邻)
     */
    void ScaleImage(
        const uint8_t* src, int src_w, int src_h,
        uint8_t* dst, int dst_w, int dst_h) {
        
        float scale_x = static_cast<float>(src_w) / dst_w;
        float scale_y = static_cast<float>(src_h) / dst_h;
        
        for (int y = 0; y < dst_h; ++y) {
            for (int x = 0; x < dst_w; ++x) {
                int src_x = static_cast<int>(x * scale_x);
                int src_y = static_cast<int>(y * scale_y);
                src_x = std::min(src_x, src_w - 1);
                src_y = std::min(src_y, src_h - 1);
                
                for (int c = 0; c < 3; ++c) {
                    dst[(y * dst_w + x) * 3 + c] = 
                        src[(src_y * src_w + src_x) * 3 + c];
                }
            }
        }
    }
};

/**
 * @brief 集成调度器 - 协调TTA和多模型融合
 */
class EnsembleScheduler {
public:
    struct EnsembleConfig {
        std::vector<std::string> model_paths;
        float tta_weight = 0.5f;
        float model_weight = 1.0f;
        int max_detections = 100;
    };
    
    explicit EnsembleScheduler(const EnsembleConfig& config) : config_(config) {}
    
    /**
     * @brief 执行集成推理
     */
    template<typename InferFunc>
    std::vector<VeDetection> Process(
        const uint8_t* image, int width, int height,
        InferFunc infer_func) {
        
        std::vector<std::vector<VeDetection>> model_results;
        
        // 1. 每个模型推理
        for (const auto& model_path : config_.model_paths) {
            auto result = infer_func(image, width, height, 
                                     model_path.c_str(), 0, 0);
            if (!result.empty()) {
                model_results.push_back(result);
            }
        }
        
        // 2. TTA增强
        TTAConfig tta_config;
        tta_config.score_threshold = 0.3f;
        tta_config.max_detections = config_.max_detections;
        tta_config.iou_threshold = 0.5f;
        
        TTAEngine tta(tta_config);
        std::vector<VeDetection> tta_result;
        
        if (config_.tta_weight > 0) {
            tta_result = tta.Process(image, width, height, infer_func);
            if (!tta_result.empty()) {
                model_results.push_back(tta_result);
            }
        }
        
        // 3. 融合所有结果
        DetectionFuser fuser;
        DetectionFuser::FuseConfig fuse_config;
        fuse_config.max_detections = config_.max_detections;
        fuse_config.iou_threshold = 0.5f;
        fuse_config.score_threshold = 0.3f;
        
        return fuser.Fuse(model_results);
    }
    
private:
    EnsembleConfig config_;
    
    // 复用DetectionFuser
    class DetectionFuser {
    public:
        struct FuseConfig {
            float iou_threshold = 0.5f;
            float score_threshold = 0.3f;
            int max_detections = 100;
            bool weighted_average = true;
        };
        
        explicit DetectionFuser(const FuseConfig& config) : config_(config) {}
        
        std::vector<VeDetection> Fuse(
            const std::vector<std::vector<VeDetection>>& detections_list) {
            
            if (detections_list.empty()) {
                return {};
            }
            
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
            
            std::sort(all_detections.begin(), all_detections.end(),
                [](const VeDetection& a, const VeDetection& b) {
                    return a.score > b.score;
                });
            
            std::vector<bool> fused(all_detections.size(), false);
            std::vector<VeDetection> result;
            
            for (size_t i = 0; i < all_detections.size(); ++i) {
                if (fused[i]) continue;
                
                std::vector<const VeDetection*> group;
                group.push_back(&all_detections[i]);
                fused[i] = true;
                
                for (size_t j = i + 1; j < all_detections.size(); ++j) {
                    if (fused[j]) continue;
                    
                    float iou = InlineIoU(all_detections[i].bbox, 
                                           all_detections[j].bbox);
                    if (iou > config_.iou_threshold) {
                        group.push_back(&all_detections[j]);
                        fused[j] = true;
                    }
                }
                
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
        
        VeDetection FuseGroup(const std::vector<const VeDetection*>& group) {
            VeDetection result = *group[0];
            
            if (config_.weighted_average) {
                float total_score = 0;
                for (const auto* det : group) {
                    total_score += det->score;
                    result.bbox.x1 += det->bbox.x1 * det->score;
                    result.bbox.y1 += det->bbox.y1 * det->score;
                    result.bbox.x2 += det->bbox.x2 * det->score;
                    result.bbox.y2 += det->bbox.y2 * det->score;
                }
                
                result.bbox.x1 /= total_score;
                result.bbox.y1 /= total_score;
                result.bbox.x2 /= total_score;
                result.bbox.y2 /= total_score;
                result.score = total_score / group.size();
            }
            
            return result;
        }
    };
};

} // namespace tta
} // namespace vision_engine

#endif // VE_TTA_H
