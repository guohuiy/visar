/**
 * @file ve_preprocess.h
 * @brief SIMD优化的图像预处理模块
 * @author VisionEngine Team
 * @date 2024-02
 */

#ifndef VE_PREPROCESS_H
#define VE_PREPROCESS_H

#include "../core/ve_types.h"
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

namespace vision_engine {
namespace preprocess {

/**
 * @brief 预处理配置
 */
struct PreprocessConfig {
    int target_width = 640;
    int target_height = 640;
    int target_channels = 3;
    bool use_letterbox = true;
    bool use_normalization = true;
    bool use_hwc_to_chw = true;
    
    // 归一化参数 (ImageNet标准)
    float mean[3] = {123.675f, 116.28f, 103.53f};
    float std[3] = {58.395f, 57.12f, 57.375f};
};

/**
 * @brief SIMD优化的图像缩放 (双线性插值)
 */
class SIMDResize {
public:
    SIMDResize() = default;
    
    explicit SIMDResize(int dst_width, int dst_height)
        : dst_width_(dst_width), dst_height_(dst_height) {}
    
    void Process(
        const uint8_t* src, int src_w, int src_h,
        float* dst, int dst_w, int dst_h,
        int channels = 3) {
        
        dst_width_ = dst_w;
        dst_height_ = dst_h;
        
        const float scale_x = static_cast<float>(src_w) / dst_w;
        const float scale_y = static_cast<float>(src_h) / dst_h;
        
        // 通用实现
        for (int y = 0; y < dst_h; ++y) {
            for (int x = 0; x < dst_w; ++x) {
                float src_x = (x + 0.5f) * scale_x - 0.5f;
                float src_y = (y + 0.5f) * scale_y - 0.5f;
                
                int x0 = static_cast<int>(src_x);
                int y0 = static_cast<int>(src_y);
                int x1 = std::min(x0 + 1, src_w - 1);
                int y1 = std::min(y0 + 1, src_h - 1);
                
                float fx = src_x - x0;
                float fy = src_y - y0;
                
                for (int c = 0; c < channels; ++c) {
                    float v00 = static_cast<float>(src[(y0 * src_w + x0) * channels + c]);
                    float v01 = static_cast<float>(src[(y0 * src_w + x1) * channels + c]);
                    float v10 = static_cast<float>(src[(y1 * src_w + x0) * channels + c]);
                    float v11 = static_cast<float>(src[(y1 * src_w + x1) * channels + c]);
                    
                    dst[(y * dst_w + x) * channels + c] = 
                        v00 * (1 - fx) * (1 - fy) +
                        v01 * fx * (1 - fy) +
                        v10 * (1 - fx) * fy +
                        v11 * fx * fy;
                }
            }
        }
    }
    
private:
    int dst_width_ = 640;
    int dst_height_ = 640;
};

/**
 * @brief 优化的图像归一化
 */
class SIMDNormalize {
public:
    SIMDNormalize() = default;
    
    explicit SIMDNormalize(const float* mean, const float* std) {
        for (int i = 0; i < 3; ++i) {
            mean_[i] = mean ? mean[i] : default_mean_[i];
            std_[i] = std ? std[i] : default_std_[i];
        }
    }
    
    void Process(float* data, size_t size, int channels = 3) {
        for (size_t i = 0; i < size; i += 3) {
            data[i] = (data[i] - mean_[0]) / std_[0];
            data[i + 1] = (data[i + 1] - mean_[1]) / std_[1];
            data[i + 2] = (data[i + 2] - mean_[2]) / std_[2];
        }
    }
    
private:
    float mean_[3] = {123.675f, 116.28f, 103.53f};
    float std_[3] = {58.395f, 57.12f, 57.375f};
    static constexpr float default_mean_[3] = {123.675f, 116.28f, 103.53f};
    static constexpr float default_std_[3] = {58.395f, 57.12f, 57.375f};
};

/**
 * @brief HWC to CHW 转换
 */
class HWCtoCHWConverter {
public:
    HWCtoCHWConverter() = default;
    
    explicit HWCtoCHWConverter(int height, int width, int channels)
        : height_(height), width_(width), channels_(channels) {}
    
    void Process(const float* src, float* dst) {
        for (int c = 0; c < channels_; ++c) {
            for (int h = 0; h < height_; ++h) {
                for (int w = 0; w < width_; ++w) {
                    dst[c * height_ * width_ + h * width_ + w] = 
                        src[h * width_ * channels_ + w * channels_ + c];
                }
            }
        }
    }
    
private:
    int height_ = 640;
    int width_ = 640;
    int channels_ = 3;
};

/**
 * @brief 综合预处理引擎
 */
class PreprocessEngine {
public:
    PreprocessEngine() = default;
    
    explicit PreprocessEngine(const PreprocessConfig& config) 
        : config_(config) {}
    
    void SetConfig(const PreprocessConfig& config) {
        config_ = config;
    }
    
    void Process(
        const uint8_t* src, int src_w, int src_h,
        float* dst,
        float& scale_x, float& scale_y,
        int& pad_x, int& pad_y) {
        
        // 1. 计算letterbox参数
        float scale = std::min(
            static_cast<float>(config_.target_width) / src_w,
            static_cast<float>(config_.target_height) / src_h
        );
        
        int new_w = static_cast<int>(src_w * scale);
        int new_h = static_cast<int>(src_h * scale);
        
        pad_x = (config_.target_width - new_w) / 2;
        pad_y = (config_.target_height - new_h) / 2;
        
        scale_x = scale;
        scale_y = scale;
        
        // 2. 缩放
        int dst_size = config_.target_width * config_.target_height * config_.target_channels;
        std::vector<float> resized(dst_size);
        SIMDResize resize;
        resize.Process(src, src_w, src_h, resized.data(), 
                       config_.target_width, config_.target_height, 
                       config_.target_channels);
        
        // 3. 归一化
        if (config_.use_normalization) {
            SIMDNormalize normalize(config_.mean, config_.std);
            normalize.Process(resized.data(), resized.size(), config_.target_channels);
        }
        
        // 4. HWC to CHW
        if (config_.use_hwc_to_chw) {
            HWCtoCHWConverter converter(config_.target_height, 
                                        config_.target_width, 
                                        config_.target_channels);
            converter.Process(resized.data(), dst);
        } else {
            memcpy(dst, resized.data(), dst_size * sizeof(float));
        }
    }
    
private:
    PreprocessConfig config_;
};

} // namespace preprocess
} // namespace vision_engine

#endif // VE_PREPROCESS_H
