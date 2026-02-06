#ifndef VE_INFERENCE_H
#define VE_INFERENCE_H

#include "ve_types.h"
#include "ve_error.h"
#include "ve_result.h"

#ifdef __cplusplus
#include <string>
#include <memory>
#include <functional>
#include <future>

namespace vision_engine {

/**
 * @brief 推理引擎主类
 */
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    /**
     * @brief 初始化引擎
     * @param options 引擎配置
     * @return 状态码
     */
    VeStatusCode Initialize(const VeEngineOptions& options);

    /**
     * @brief 加载模型
     * @param model_path 模型路径
     * @return 状态码
     */
    VeStatusCode LoadModel(const std::string& model_path);

    /**
     * @brief 卸载模型
     */
    void UnloadModel();

    /**
     * @brief 检查模型是否已加载
     */
    bool IsModelLoaded() const;

    /**
     * @brief 推理单张图像
     * @param image 图像数据
     * @return 推理结果
     */
    std::shared_ptr<InferenceResult> Infer(const VeImageData& image);

    /**
     * @brief 推理图像数据指针
     * @param data 图像数据指针
     * @param width 宽度
     * @param height 高度
     * @param format 图像格式
     * @return 推理结果
     */
    std::shared_ptr<InferenceResult> Infer(const uint8_t* data,
                                            int32_t width,
                                            int32_t height,
                                            VeImageFormat format);

    /**
     * @brief 异步推理
     * @param image 图像数据
     * @return Future对象
     */
    std::future<std::shared_ptr<InferenceResult>> InferAsync(const VeImageData& image);

    /**
     * @brief 设置推理完成回调
     */
    void SetResultCallback(ResultCallback callback);

    /**
     * @brief 批量推理
     * @param images 图像数组
     * @param batch_size 批大小
     * @return 批处理结果
     */
    std::shared_ptr<BatchResult> InferBatch(const VeImageData* images, 
                                            int32_t batch_size);

    /**
     * @brief 预热模型
     * @return 状态码
     */
    VeStatusCode Warmup();

    /**
     * @brief 获取模型信息
     */
    VeModelInfo GetModelInfo() const;

    /**
     * @brief 获取性能指标
     */
    VePerformanceMetrics GetPerformanceMetrics() const;

    /**
     * @brief 设置置信度阈值
     */
    void SetConfidenceThreshold(float threshold);

    /**
     * @brief 设置NMS阈值
     */
    void SetNMSThreshold(float threshold);

    /**
     * @brief 获取最后错误
     */
    std::string GetLastError() const;

    /**
     * @brief 获取支持的设备列表
     */
    std::vector<VeDeviceType> GetSupportedDevices() const;

    /**
     * @brief 获取支持的精度列表
     */
    std::vector<VePrecision> GetSupportedPrecisions(VeDeviceType device) const;

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
 * @brief 创建推理引擎实例
 */
VE_API VeInferenceHandle ve_inference_create(const VeEngineOptions* options);

/**
 * @brief 销毁推理引擎实例
 */
VE_API void ve_inference_destroy(VeInferenceHandle handle);

/**
 * @brief 加载模型
 */
VE_API VeStatusCode ve_inference_load_model(VeInferenceHandle handle, 
                                              const char* model_path);

/**
 * @brief 卸载模型
 */
VE_API void ve_inference_unload_model(VeInferenceHandle handle);

/**
 * @brief 检查模型是否加载
 */
VE_API bool ve_inference_is_model_loaded(VeInferenceHandle handle);

/**
 * @brief 推理图像
 */
VE_API VeInferenceResult* ve_inference_infer(VeInferenceHandle handle,
                                               const VeImageData* image);

/**
 * @brief 异步推理
 */
VE_API void ve_inference_infer_async(VeInferenceHandle handle,
                                      const VeImageData* image,
                                      VeInferenceCallback callback);

/**
 * @brief 设置结果回调
 */
VE_API void ve_inference_set_callback(VeInferenceHandle handle,
                                       VeInferenceCallback callback);

/**
 * @brief 批量推理
 */
VE_API VeInferenceResult** ve_inference_infer_batch(VeInferenceHandle handle,
                                                     const VeImageData* images,
                                                     int32_t batch_size,
                                                     int32_t* result_count);

/**
 * @brief 预热模型
 */
VE_API VeStatusCode ve_inference_warmup(VeInferenceHandle handle);

/**
 * @brief 获取性能指标
 */
VE_API VePerformanceMetrics ve_inference_get_metrics(VeInferenceHandle handle);

#ifdef __cplusplus
}
#endif

#endif // VE_INFERENCE_H
