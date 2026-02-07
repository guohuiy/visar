#ifndef VE_TENSORRT_BACKEND_H
#define VE_TENSORRT_BACKEND_H

#include "../core/ve_types.h"
#include "../core/ve_error.h"

#ifdef __cplusplus
#include <string>
#include <memory>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace vision_engine {

/**
 * @brief TensorRT 后端实现
 * 提供NVIDIA GPU加速推理
 */
class TensorRTBackend {
public:
    TensorRTBackend();
    ~TensorRTBackend();

    /**
     * @brief 初始化后端
     * @param model_path 模型路径 (.onnx或.trt)
     * @param options 配置选项
     * @return 状态码
     */
    VeStatusCode Initialize(const std::string& model_path, const VeEngineOptions& options);

    /**
     * @brief 推理 (FP32/FP16)
     * @param input_data 输入数据
     * @param input_shape 输入形状
     * @param output_data 输出数据
     * @return 状态码
     */
    VeStatusCode Infer(const float* input_data,
                       const std::vector<int64_t>& input_shape,
                       float* output_data);

    /**
     * @brief 推理 (INT8量化)
     * @param input_data 输入数据 (uint8_t)
     * @param input_shape 输入形状
     * @param output_data 输出数据
     * @return 状态码
     */
    VeStatusCode InferINT8(const uint8_t* input_data,
                          const std::vector<int64_t>& input_shape,
                          float* output_data);

    /**
     * @brief 获取输入节点名称
     */
    std::vector<std::string> GetInputNames() const;

    /**
     * @brief 获取输出节点名称
     */
    std::vector<std::string> GetOutputNames() const;

    /**
     * @brief 获取输入形状
     */
    std::vector<int64_t> GetInputShape() const;

    /**
     * @brief 获取输出形状
     */
    std::vector<int64_t> GetOutputShape() const;

    /**
     * @brief 启用FP16精度
     */
    void EnableFP16();

    /**
     * @brief 启用INT8量化
     * @param calib_data 校准数据
     */
    void EnableINT8(const std::vector<std::vector<float>>& calib_data);

    /**
     * @brief 设置工作空间大小
     */
    void SetWorkspaceSize(size_t size_mb);

    /**
     * @brief 获取GPU内存使用量
     */
    size_t GetDeviceMemoryUsage() const;

    /**
     * @brief 预热模型
     */
    void Warmup(int iterations = 10);

    /**
     * @brief 获取引擎信息
     */
    std::string GetEngineInfo() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_engine

#endif // __cplusplus

#endif // VE_TENSORRT_BACKEND_H
