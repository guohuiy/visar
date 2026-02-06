#ifndef VE_QUANTIZATION_H
#define VE_QUANTIZATION_H

#include "ve_types.h"
#include "ve_error.h"

#ifdef __cplusplus
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace vision_engine {

/**
 * @brief 量化类型
 */
enum class QuantizationType {
    PTQ = 0,      // 后训练量化 (Post-Training Quantization)
    QAT,          // 量化感知训练 (Quantization-Aware Training)
    Dynamic       // 动态量化
};

/**
 * @brief 校准方法
 */
enum class CalibrationMethod {
    MinMax = 0,    // 最小最大值
    Entropy,       // 熵方法
    Percentile     // 百分位方法
};

/**
 * @brief 量化配置
 */
struct QuantConfig {
    QuantizationType type = QuantizationType::PTQ;
    CalibrationMethod calibMethod = CalibrationMethod::MinMax;
    int calibrationSamples = 500;    // 校准数据集大小
    bool mixedPrecision = false;      // 是否启用混合精度
    std::vector<std::string> fp16Layers;  // 保留FP16的层
    
    // 静态方法：移动端量化策略
    static QuantConfig ForMobile() {
        return {
            QuantizationType::PTQ,
            CalibrationMethod::MinMax,
            500,
            true,
            {}
        };
    }
    
    // 静态方法：服务器GPU量化策略
    static QuantConfig ForServerGPU() {
        return {
            QuantizationType::PTQ,
            CalibrationMethod::Entropy,
            1000,
            false,
            {}
        };
    }
};

/**
 * @brief 量化模型评估指标
 */
struct QuantizationMetrics {
    float accuracyDrop = 0.0f;         // 精度损失
    float inferenceSpeedup = 1.0f;     // 推理加速比
    size_t modelSizeReduction = 0;     // 模型大小减少量
    float compressionRatio = 1.0f;     // 压缩比
};

/**
 * @brief INT8量化引擎
 */
class QuantizationEngine {
public:
    QuantizationEngine();
    ~QuantizationEngine();

    /**
     * @brief PTQ后训练量化
     * @param fp32Model FP32模型路径
     * @param calibDataPath 校准数据路径
     * @param config 量化配置
     * @param outputPath 输出路径
     * @return 状态码
     */
    VeStatusCode QuantizePTQ(const std::string& fp32Model,
                            const std::string& calibDataPath,
                            const QuantConfig& config,
                            std::string& outputPath);

    /**
     * @brief 动态量化 (无需校准数据)
     * @param fp32Model FP32模型路径
     * @param outputPath 输出路径
     * @return 状态码
     */
    VeStatusCode QuantizeDynamic(const std::string& fp32Model,
                                 std::string& outputPath);

    /**
     * @brief QAT模型转换 (需要训练框架集成)
     * @param qatModel QAT模型路径
     * @param outputPath 输出路径
     * @return 状态码
     */
    VeStatusCode ConvertQAT(const std::string& qatModel,
                           std::string& outputPath);

    /**
     * @brief 混合精度优化
     * @param modelPath 模型路径
     * @param sensitivityProfile 敏感度配置
     * @return 状态码
     */
    VeStatusCode OptimizeMixedPrecision(std::string& modelPath,
                                        const std::string& sensitivityProfile);

    /**
     * @brief 评估量化模型
     * @param modelPath 模型路径
     * @param testDataPath 测试数据路径
     * @return 评估指标
     */
    QuantizationMetrics EvaluateQuantModel(const std::string& modelPath,
                                            const std::string& testDataPath);

    /**
     * @brief 生成校准数据
     * @param imagePaths 图像路径列表
     * @param outputPath 输出路径
     * @param numSamples 采样数量
     * @return 状态码
     */
    VeStatusCode GenerateCalibrationData(const std::vector<std::string>& imagePaths,
                                         const std::string& outputPath,
                                         int numSamples = 500);

    /**
     * @brief 分析模型敏感度
     * @param modelPath 模型路径
     * @param testDataPath 测试数据路径
     * @return 敏感度报告
     */
    std::string AnalyzeSensitivity(const std::string& modelPath,
                                   const std::string& testDataPath);

    /**
     * @brief 设置校准数据目录
     */
    void SetCalibrationDataPath(const std::string& path);

    /**
     * @brief 设置进度回调
     */
    void SetProgressCallback(std::function<void(float, const std::string&)> callback);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 量化工具命令行接口
 */
class QuantizationCLI {
public:
    static int Run(int argc, char** argv);
};

} // namespace vision_engine

#endif // __cplusplus

// C API
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建量化引擎
 */
VE_API void* ve_quantization_create();

/**
 * @brief 销毁量化引擎
 */
VE_API void ve_quantization_destroy(void* handle);

/**
 * @brief PTQ量化
 */
VE_API VeStatusCode ve_quantization_ptq(void* handle,
                                         const char* model_path,
                                         const char* calib_data_path,
                                         const VeQuantConfig* config,
                                         char* output_path,
                                         int output_path_size);

/**
 * @brief 动态量化
 */
VE_API VeStatusCode ve_quantization_dynamic(void* handle,
                                              const char* model_path,
                                              char* output_path,
                                              int output_path_size);

/**
 * @brief 评估量化模型
 */
VE_API VeQuantizationMetrics ve_quantization_evaluate(void* handle,
                                                       const char* model_path,
                                                       const char* test_data_path);

#ifdef __cplusplus
}
#endif

#endif // VE_QUANTIZATION_H
