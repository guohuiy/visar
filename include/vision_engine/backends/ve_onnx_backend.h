#ifndef VE_ONNX_BACKEND_H
#define VE_ONNX_BACKEND_H

#include "ve_types.h"
#include "ve_error.h"

#ifdef __cplusplus
#include <string>
#include <memory>
#include <vector>

// 前向声明 ONNX Runtime 类型
namespace Ort {
    class Env;
    class Session;
    class SessionOptions;
}

namespace vision_engine {

/**
 * @brief ONNX Runtime 后端实现
 */
class ONNXBackend {
public:
    ONNXBackend();
    ~ONNXBackend();

    /**
     * @brief 初始化后端
     * @param model_path 模型路径
     * @return 状态码
     */
    VeStatusCode Initialize(const std::string& model_path);

    /**
     * @brief 推理
     * @param input_data 输入数据
     * @param input_shape 输入形状
     * @param output_data 输出数据
     * @return 状态码
     */
    VeStatusCode Infer(const float* input_data,
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
     * @brief 设置执行providers
     */
    void SetExecutionProviders(const std::vector<std::string>& providers);

    /**
     * @brief 启用CUDA
     */
    void EnableCUDA();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_engine

#endif // __cplusplus

#endif // VE_ONNX_BACKEND_H
