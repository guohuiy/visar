#include "ve_options.h"
#include <algorithm>

namespace vision_engine {

EngineOptions::EngineOptions() = default;

EngineOptions::~EngineOptions() = default;

void EngineOptions::SetPreferredBackend(VeBackendType backend) {
    preferred_backend_ = backend;
}

VeBackendType EngineOptions::GetPreferredBackend() const {
    return preferred_backend_;
}

void EngineOptions::SetDeviceType(VeDeviceType device) {
    device_type_ = device;
}

VeDeviceType EngineOptions::GetDeviceType() const {
    return device_type_;
}

void EngineOptions::SetPrecision(VePrecision precision) {
    precision_ = precision;
}

VePrecision EngineOptions::GetPrecision() const {
    return precision_;
}

void EngineOptions::SetGPUId(int32_t gpu_id) {
    gpu_id_ = gpu_id;
}

int32_t EngineOptions::GetGPUId() const {
    return gpu_id_;
}

void EngineOptions::SetNumThreads(int32_t num_threads) {
    num_threads_ = std::max(1, num_threads);
}

int32_t EngineOptions::GetNumThreads() const {
    return num_threads_;
}

void EngineOptions::SetBatchSize(int32_t batch_size) {
    batch_size_ = std::max(1, batch_size);
}

int32_t EngineOptions::GetBatchSize() const {
    return batch_size_;
}

void EngineOptions::SetConfidenceThreshold(float threshold) {
    confidence_threshold_ = std::clamp(threshold, 0.0f, 1.0f);
}

float EngineOptions::GetConfidenceThreshold() const {
    return confidence_threshold_;
}

void EngineOptions::SetNMSThreshold(float threshold) {
    nms_threshold_ = std::clamp(threshold, 0.0f, 1.0f);
}

float EngineOptions::GetNMSThreshold() const {
    return nms_threshold_;
}

void EngineOptions::SetModelPath(const std::string& path) {
    model_path_ = path;
}

std::string EngineOptions::GetModelPath() const {
    return model_path_;
}

void EngineOptions::SetConfigPath(const std::string& path) {
    config_path_ = path;
}

std::string EngineOptions::GetConfigPath() const {
    return config_path_;
}

void EngineOptions::SetCachePath(const std::string& path) {
    cache_path_ = path;
}

std::string EngineOptions::GetCachePath() const {
    return cache_path_;
}

void EngineOptions::EnableProfiling(bool enable) {
    enable_profiling_ = enable;
}

bool EngineOptions::IsProfilingEnabled() const {
    return enable_profiling_;
}

void EngineOptions::EnableDebug(bool enable) {
    enable_debug_ = enable;
}

bool EngineOptions::IsDebugEnabled() const {
    return enable_debug_;
}

void EngineOptions::SetQuantizationType(VePrecision type) {
    quantization_type_ = type;
}

VePrecision EngineOptions::GetQuantizationType() const {
    return quantization_type_;
}

void EngineOptions::SetCalibrationSamples(int32_t samples) {
    calibration_samples_ = std::max(1, samples);
}

int32_t EngineOptions::GetCalibrationSamples() const {
    return calibration_samples_;
}

void EngineOptions::SetOTAServerURL(const std::string& url) {
    ota_server_url_ = url;
}

std::string EngineOptions::GetOTAServerURL() const {
    return ota_server_url_;
}

void EngineOptions::EnableAutoUpdate(bool enable) {
    auto_update_ = enable;
}

bool EngineOptions::IsAutoUpdateEnabled() const {
    return auto_update_;
}

VeEngineOptions EngineOptions::ToCStruct() const {
    VeEngineOptions opt;
    opt.preferred_backend = preferred_backend_;
    opt.device_type = device_type_;
    opt.precision = precision_;
    opt.num_threads = num_threads_;
    opt.enable_profiling = enable_profiling_;
    opt.enable_debug = enable_debug_;
    opt.model_path = model_path_.empty() ? nullptr : model_path_.c_str();
    opt.config_path = config_path_.empty() ? nullptr : config_path_.c_str();
    opt.cache_path = cache_path_.empty() ? nullptr : cache_path_.c_str();
    opt.batch_size = batch_size_;
    opt.gpu_id = gpu_id_;
    return opt;
}

} // namespace vision_engine
