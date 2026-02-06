#ifndef VE_RESULT_H
#define VE_RESULT_H

#include "ve_types.h"
#include "ve_error.h"

#ifdef __cplusplus
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace vision_engine {

/**
 * @brief 推理结果类
 */
class InferenceResult {
public:
    InferenceResult();
    ~InferenceResult();

    /**
     * @brief 获取检测结果数量
     */
    int32_t GetDetectionCount() const;

    /**
     * @brief 获取指定索引的检测结果
     */
    VeDetection GetDetection(int32_t index) const;

    /**
     * @brief 获取所有检测结果
     */
    const VeDetection* GetDetections() const;

    /**
     * @brief 获取推理耗时 (毫秒)
     */
    double GetInferenceTimeMs() const;

    /**
     * @brief 获取原始输出
     */
    const void* GetRawOutput() const;

    /**
     * @brief 获取原始输出大小
     */
    size_t GetRawOutputSize() const;

    /**
     * @brief 获取性能指标
     */
    VePerformanceMetrics GetPerformanceMetrics() const;

    /**
     * @brief 获取JSON格式结果
     */
    std::string ToJSON() const;

    /**
     * @brief 转换为可视化结果
     * @param original_image 原始图像
     * @return 绘制了检测框的图像
     */
    std::vector<uint8_t> Visualize(const std::vector<uint8_t>& original_image,
                                    int32_t width, int32_t height) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 结果回调
 */
using ResultCallback = std::function<void(std::shared_ptr<InferenceResult>)>;

/**
 * @brief 批处理结果
 */
class BatchResult {
public:
    BatchResult();
    ~BatchResult();

    /**
     * @brief 获取批大小
     */
    int32_t GetBatchSize() const;

    /**
     * @brief 获取指定批次的推理结果
     */
    std::shared_ptr<InferenceResult> GetResult(int32_t index) const;

    /**
     * @brief 添加推理结果
     */
    void AddResult(std::shared_ptr<InferenceResult> result);

    /**
     * @brief 获取平均推理时间
     */
    double GetAverageInferenceTimeMs() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_engine

#endif // __cplusplus

// C API
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 获取检测结果数量
 */
VE_API int32_t ve_result_get_detection_count(const VeInferenceResult* result);

/**
 * @brief 获取检测结果数组
 */
VE_API const VeDetection* ve_result_get_detections(const VeInferenceResult* result);

/**
 * @brief 获取推理时间
 */
VE_API float ve_result_get_inference_time(const VeInferenceResult* result);

/**
 * @brief 释放推理结果
 */
VE_API void ve_result_destroy(VeInferenceResult* result);

#ifdef __cplusplus
}
#endif

#endif // VE_RESULT_H
