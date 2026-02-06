#ifndef VE_OPTIONS_H
#define VE_OPTIONS_H

#include "ve_types.h"

#ifdef __cplusplus
#include <string>
#include <vector>

namespace vision_engine {

/**
 * @brief VisionEngine 配置选项类 (C++)
 */
class EngineOptions {
public:
    EngineOptions();
    ~EngineOptions();

    // 后端配置
    void SetPreferredBackend(VeBackendType backend);
    VeBackendType GetPreferredBackend() const;

    void SetDeviceType(VeDeviceType device);
    VeDeviceType GetDeviceType() const;

    void SetPrecision(VePrecision precision);
    VePrecision GetPrecision() const;

    void SetGPUId(int32_t gpu_id);
    int32_t GetGPUId() const;

    // 推理配置
    void SetNumThreads(int32_t num_threads);
    int32_t GetNumThreads() const;

    void SetBatchSize(int32_t batch_size);
    int32_t GetBatchSize() const;

    void SetConfidenceThreshold(float threshold);
    float GetConfidenceThreshold() const;

    void SetNMSThreshold(float threshold);
    float GetNMSThreshold() const;

    // 路径配置
    void SetModelPath(const std::string& path);
    std::string GetModelPath() const;

    void SetConfigPath(const std::string& path);
    std::string GetConfigPath() const;

    void SetCachePath(const std::string& path);
    std::string GetCachePath() const;

    // 调试配置
    void EnableProfiling(bool enable);
    bool IsProfilingEnabled() const;

    void EnableDebug(bool enable);
    bool IsDebugEnabled() const;

    // 量化配置
    void SetQuantizationType(VePrecision type);
    VePrecision GetQuantizationType() const;

    void SetCalibrationSamples(int32_t samples);
    int32_t GetCalibrationSamples() const;

    // OTA配置
    void SetOTAServerURL(const std::string& url);
    std::string GetOTAServerURL() const;

    void EnableAutoUpdate(bool enable);
    bool IsAutoUpdateEnabled() const;

    // 转换为C结构
    VeEngineOptions ToCStruct() const;

private:
    VeBackendType preferred_backend_ = VE_BACKEND_ONNX;
    VeDeviceType device_type_ = VE_DEVICE_CPU;
    VePrecision precision_ = VE_PRECISION_FP32;
    int32_t gpu_id_ = 0;

    int32_t num_threads_ = 4;
    int32_t batch_size_ = 1;
    float confidence_threshold_ = 0.5f;
    float nms_threshold_ = 0.45f;

    std::string model_path_;
    std::string config_path_;
    std::string cache_path_;

    bool enable_profiling_ = false;
    bool enable_debug_ = false;

    VePrecision quantization_type_ = VE_PRECISION_FP32;
    int32_t calibration_samples_ = 100;

    std::string ota_server_url_;
    bool auto_update_ = false;
};

} // namespace vision_engine

#endif // __cplusplus

#endif // VE_OPTIONS_H
