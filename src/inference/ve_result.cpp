#include "ve_result.h"
#include <algorithm>
#include <numeric>
#include <sstream>

namespace vision_engine {

class InferenceResult::Impl {
public:
    std::vector<VeDetection> detections_;
    double inference_time_ms_ = 0;
    std::vector<float> raw_output_;
    VePerformanceMetrics metrics_;
};

InferenceResult::InferenceResult() : impl_(std::make_unique<Impl>()) {}

InferenceResult::~InferenceResult() = default;

int32_t InferenceResult::GetDetectionCount() const {
    return static_cast<int32_t>(impl_->detections_.size());
}

VeDetection InferenceResult::GetDetection(int32_t index) const {
    if (index >= 0 && index < static_cast<int32_t>(impl_->detections_.size())) {
        return impl_->detections_[index];
    }
    return VeDetection{};
}

const VeDetection* InferenceResult::GetDetections() const {
    return impl_->detections_.data();
}

double InferenceResult::GetInferenceTimeMs() const {
    return impl_->inference_time_ms_;
}

const void* InferenceResult::GetRawOutput() const {
    return impl_->raw_output_.data();
}

size_t InferenceResult::GetRawOutputSize() const {
    return impl_->raw_output_.size() * sizeof(float);
}

VePerformanceMetrics InferenceResult::GetPerformanceMetrics() const {
    return impl_->metrics_;
}

std::string InferenceResult::ToJSON() const {
    std::ostringstream oss;
    oss << "{\"detections\":[";
    for (size_t i = 0; i < impl_->detections_.size(); ++i) {
        const auto& det = impl_->detections_[i];
        if (i > 0) oss << ",";
        oss << "{\"class_id\":" << det.class_id
            << ",\"score\":" << det.score
            << ",\"bbox\":[" << det.bbox.x1 << "," << det.bbox.y1 
            << "," << det.bbox.x2 << "," << det.bbox.y2 << "]}";
    }
    oss << "],\"inference_time_ms\":" << impl_->inference_time_ms_ << "}";
    return oss.str();
}

std::vector<uint8_t> InferenceResult::Visualize(const std::vector<uint8_t>& original_image,
                                                  int32_t width, int32_t height) const {
    // 简化实现：返回原始图像
    return original_image;
}

class BatchResult::Impl {
public:
    std::vector<std::shared_ptr<InferenceResult>> results_;
};

BatchResult::BatchResult() : impl_(std::make_unique<Impl>()) {}

BatchResult::~BatchResult() = default;

int32_t BatchResult::GetBatchSize() const {
    return static_cast<int32_t>(impl_->results_.size());
}

std::shared_ptr<InferenceResult> BatchResult::GetResult(int32_t index) const {
    if (index >= 0 && index < static_cast<int32_t>(impl_->results_.size())) {
        return impl_->results_[index];
    }
    return nullptr;
}

void BatchResult::AddResult(std::shared_ptr<InferenceResult> result) {
    impl_->results_.push_back(result);
}

double BatchResult::GetAverageInferenceTimeMs() const {
    if (impl_->results_.empty()) return 0;
    double sum = std::accumulate(impl_->results_.begin(), impl_->results_.end(), 0.0,
        [](double acc, const auto& r) { return acc + r->GetInferenceTimeMs(); });
    return sum / impl_->results_.size();
}

// C API实现
int32_t ve_result_get_detection_count(const VeInferenceResult* result) {
    if (result) {
        return result->num_detections;
    }
    return 0;
}

const VeDetection* ve_result_get_detections(const VeInferenceResult* result) {
    if (result) {
        return result->detections;
    }
    return nullptr;
}

float ve_result_get_inference_time(const VeInferenceResult* result) {
    if (result) {
        return result->inference_time_ms;
    }
    return 0;
}

void ve_result_destroy(VeInferenceResult* result) {
    if (result) {
        if (result->detections) {
            delete[] result->detections;
        }
        delete result;
    }
}

} // namespace vision_engine
