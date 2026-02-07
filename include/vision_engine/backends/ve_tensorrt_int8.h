/**
 * @file ve_tensorrt_int8.h
 * @brief TensorRT INT8 量化推理模块
 * @author VisionEngine Team
 * @date 2024-02
 */

#ifndef VE_TENSORRT_INT8_H
#define VE_TENSORRT_INT8_H

#include "../core/ve_types.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#endif

namespace vision_engine {

/**
 * @brief INT8 校准器配置
 */
struct INT8CalibratorConfig {
    // 校准参数
    std::string calibration_data_path = "./calibration_data";  // 校准数据路径
    int batch_size = 1;                                       // 批大小
    int calibration_steps = 100;                             // 校准步数
    float quantile = 0.9999f;                                 // 分位数
    bool use_quantizer = true;                                // 使用外部量化器
    
    // 精度阈值
    float max_accuracy_drop = 0.01f;                         // 最大精度损失
};

/**
 * @brief INT8 量化配置
 */
struct INT8QuantConfig {
    // 量化参数
    bool enable_int8 = true;                                 // 启用INT8
    bool enable_fp16 = false;                                // 启用FP16混合精度
    bool use_dla = false;                                     // 使用DLA
    int dla_core = -1;                                       // DLA核心ID
    
    // INT8参数
    bool use_int8_calibrator = true;                         // 使用校准器
    int max_batch_size = 8;                                  // 最大批大小
    int max_workspace_size = 512 * 1024 * 1024;             // 最大工作空间 (512MB)
    
    // 混合精度层
    std::vector<std::string> fp16_layers;                    // 保留FP16的层
    std::vector<std::string> int8_layers;                    // 强制INT8的层
};

/**
 * @brief TensorRT INT8 引擎构建器
 */
class TensorRTINT8Builder {
public:
    TensorRTINT8Builder();
    ~TensorRTINT8Builder();
    
    /**
     * @brief 加载ONNX模型
     */
    bool LoadONNX(const std::string& model_path);
    
    /**
     * @brief 配置INT8量化
     */
    void ConfigureINT8(const INT8QuantConfig& config);
    
    /**
     * @brief 配置FP16混合精度
     */
    void ConfigureFP16(const std::vector<std::string>& fp16_layers = {});
    
    /**
     * @brief 设置校准数据
     */
    void SetCalibrationData(const std::vector<std::vector<float>>& data);
    
    /**
     * @brief 设置图像校准数据
     */
    void SetImageCalibrationData(
        const std::vector<std::vector<uint8_t>>& images,
        int input_width,
        int input_height);
    
    /**
     * @brief 构建引擎
     */
    bool BuildEngine(const std::string& output_path);
    
    /**
     * @brief 获取模型信息
     */
    struct ModelInfo {
        int input_width;
        int input_height;
        int input_channels;
        int num_classes;
        std::string precision;
    };
    ModelInfo GetModelInfo() const;
    
private:
#ifdef HAVE_TENSORRT
    std::unique_ptr<nvinfer1::IBuilder> builder_;
    std::unique_ptr<nvinfer1::INetworkDefinition> network_;
    std::unique_ptr<nvinfer1::IBuilderConfig> config_;
    std::unique_ptr<nvonnxparser::IParser> parser_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
#endif
    
    INT8QuantConfig quant_config_;
    std::vector<std::vector<float>> calibration_data_;
    std::vector<std::vector<uint8_t>> calibration_images_;
    int cal_input_width_ = 640;
    int cal_input_height_ = 640;
    ModelInfo model_info_;
    bool onnx_loaded_ = false;
};

/**
 * @brief INT8 校准器 (使用TensorRT内置)
 */
#ifdef HAVE_TENSORRT
class INT8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    INT8EntropyCalibrator(
        const std::vector<std::vector<float>>& data,
        const std::string& input_name,
        const std::vector<int64_t>& input_shape,
        int batch_size);
    
    ~INT8EntropyCalibrator() override;
    
    /**
     * @brief 获取批次数据
     */
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
    
    /**
     * @brief 获取校准器名称
     */
    const char* getName() const override;
    
    /**
     * @brief 读取校准缓存
     */
    int readCalibrationCache(uint8_t* cache, size_t size) override;
    
    /**
     * @brief 写入校准缓存
     */
    void writeCalibrationCache(const uint8_t* cache, size_t size) override;
    
private:
    std::vector<std::vector<float>> data_;
    std::string input_name_;
    std::vector<int64_t> input_shape_;
    int batch_size_;
    size_t current_batch_ = 0;
    std::string calibrator_name_ = "INT8EntropyCalibrator";
};
#endif

/**
 * @brief 图像预处理校准器
 */
class ImageCalibrator {
public:
    ImageCalibrator(
        const std::vector<std::vector<uint8_t>>& images,
        int input_width,
        int input_height,
        float mean[3] = {123.675f, 116.28f, 103.53f},
        float std[3] = {58.395f, 57.12f, 57.375f});
    
    /**
     * @brief 获取预处理后的校准数据
     */
    std::vector<std::vector<float>> GetPreprocessedData() const;
    
    /**
     * @brief 获取批次数据
     */
    std::vector<float> GetBatch(int batch_idx, int batch_size) const;
    
    /**
     * @brief 获取总批次数
     */
    int GetBatchCount() const;
    
private:
    std::vector<std::vector<uint8_t>> images_;
    int input_width_;
    int input_height_;
    float mean_[3];
    float std_[3];
};

/**
 * @brief INT8 量化评估器
 */
class INT8Quantizer {
public:
    struct QuantizationResult {
        float accuracy_drop;           // 精度损失
        float compression_ratio;       // 压缩比
        float speedup;                 // 加速比
        size_t original_size;          // 原始模型大小
        size_t quantized_size;         // 量化后模型大小
        bool success;                  // 是否成功
        std::string message;           // 状态信息
    };
    
    /**
     * @brief 量化模型并评估
     */
    QuantizationResult QuantizeAndEvaluate(
        const std::string& onnx_path,
        const std::vector<std::vector<uint8_t>>& calibration_images,
        const std::string& output_path,
        float confidence_threshold = 0.5f);
    
    /**
     * @brief 计算压缩比
     */
    static float CalculateCompressionRatio(
        const std::string& original_path,
        const std::string& quantized_path);
    
    /**
     * @brief 评估量化模型精度
     */
    static float EvaluatePrecision(
        const std::string& model_path,
        const std::vector<std::vector<uint8_t>>& test_images,
        const std::vector<std::vector<int>>& ground_truth);
    
private:
    INT8CalibratorConfig config_;
};

/**
 * @brief 混合精度推理引擎
 */
class MixedPrecisionEngine {
public:
    MixedPrecisionEngine();
    
    /**
     * @brief 配置混合精度
     */
    void Configure(
        const std::vector<std::string>& fp16_layers,
        const std::vector<std::string>& int8_layers);
    
    /**
     * @brief 获取最优精度策略
     */
    std::vector<std::string> GetOptimalPrecision(
        const std::string& model_path,
        float accuracy_threshold = 0.98f);
    
    /**
     * @brief 敏感层分析
     */
    std::vector<std::pair<std::string, float>> AnalyzeSensitivity(
        const std::string& model_path,
        const std::vector<std::vector<uint8_t>>& test_images);
    
private:
    std::vector<std::string> fp16_layers_;
    std::vector<std::string> int8_layers_;
};

} // namespace vision_engine

#endif // VE_TENSORRT_INT8_H
