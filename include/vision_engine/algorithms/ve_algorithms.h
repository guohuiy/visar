#ifndef VE_ALGORITHMS_H
#define VE_ALGORITHMS_H

#include "../core/ve_types.h"
#include "../core/ve_error.h"

#ifdef __cplusplus
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace vision_engine {

/**
 * @brief 目标检测算法基类
 */
class ObjectDetector {
public:
    struct Config {
        float confidence_threshold = 0.5f;
        float nms_threshold = 0.45f;
        int max_detections = 100;
        bool use_letterbox = true;
    };
    
    struct Detection {
        int class_id;
        float score;
        VeBBox bbox;
        std::vector<VeKeyPoint> keypoints;
        float* mask = nullptr;
        int mask_width = 0;
        int mask_height = 0;
    };
    
    virtual ~ObjectDetector() = default;
    
    virtual VeStatusCode Initialize(const std::string& model_path, const Config& config);
    virtual std::vector<Detection> Detect(const uint8_t* image_data, 
                                          int width, int height, 
                                          VeImageFormat format) = 0;
    virtual std::vector<Detection> Detect(const VeImageData& image);
    
    void SetConfidenceThreshold(float threshold);
    void SetNMSThreshold(float threshold);
    
protected:
    Config config_;
};

/**
 * @brief YOLO目标检测器
 */
class YOLODetector : public ObjectDetector {
public:
    enum class Version {
        YOLOv5,
        YOLOv8,
        YOLOv9,
        YOLOX
    };
    
    YOLODetector();
    virtual ~YOLODetector() = default;
    
    VeStatusCode Initialize(const std::string& model_path, const Config& config) override;
    std::vector<Detection> Detect(const uint8_t* image_data,
                                  int width, int height,
                                  VeImageFormat format) override;
    
    void SetModelVersion(Version version);
    
private:
    Version version_ = Version::YOLOv8;
    std::vector<std::string> class_names_;
    
    void Preprocess(const uint8_t* image, int width, int height, float* output);
    std::vector<Detection> Postprocess(const float* output, int output_size, int original_width, int original_height,
                                        float scale_x, float scale_y, int pad_x, int pad_y);
};

/**
 * @brief 图像分割算法基类
 */
class ImageSegmenter {
public:
    struct Config {
        float threshold = 0.5f;
        bool return_probs = false;
        int output_height = 0;  // 0 means same as input
        int output_width = 0;
    };
    
    struct SegmentationResult {
        uint8_t* mask_data = nullptr;
        int width = 0;
        int height = 0;
        int num_classes = 0;
        float* class_probs = nullptr;
    };
    
    virtual ~ImageSegmenter() = default;
    virtual SegmentationResult Segment(const uint8_t* image_data, int width, int height) = 0;
    
protected:
    Config config_;
};

/**
 * @brief UNet分割器
 */
class UNetSegmenter : public ImageSegmenter {
public:
    VeStatusCode Initialize(const std::string& model_path, const Config& config);
    SegmentationResult Segment(const uint8_t* image_data, int width, int height) override;
};

/**
 * @brief DeepLab分割器
 */
class DeepLabSegmenter : public ImageSegmenter {
public:
    VeStatusCode Initialize(const std::string& model_path, const Config& config);
    SegmentationResult Segment(const uint8_t* image_data, int width, int height) override;
};

/**
 * @brief OCR识别器
 */
class OCRRecognizer {
public:
    struct Config {
        float confidence_threshold = 0.3f;
        bool detect_direction = true;
        std::string language = "ch+en";
    };
    
    struct TextDetection {
        VeBBox bbox;
        float score;
        std::vector<VeBBox> text_lines;
    };
    
    struct TextRecognition {
        std::string text;
        float score;
        VeBBox bbox;
    };
    
    virtual ~OCRRecognizer() = default;
    
    virtual VeStatusCode Initialize(const std::string& det_model_path, 
                                    const std::string& rec_model_path,
                                    const Config& config);
    
    virtual std::vector<TextDetection> DetectText(const uint8_t* image_data, 
                                                   int width, int height);
    
    virtual std::vector<TextRecognition> Recognize(const uint8_t* image_data,
                                                     int width, int height);
    
    std::vector<TextDetection> DetectAndRecognize(const uint8_t* image_data,
                                                   int width, int height);
    
protected:
    Config config_;
    std::unique_ptr<ObjectDetector> detector_;
};

/**
 * @brief 算法插件管理器
 */
class AlgorithmPluginManager {
public:
    static AlgorithmPluginManager& Instance();
    
    template<typename T>
    void RegisterAlgorithm(const std::string& name) {
        // 注册算法插件
    }
    
    std::unique_ptr<ObjectDetector> CreateDetector(const std::string& name);
    std::unique_ptr<ImageSegmenter> CreateSegmenter(const std::string& name);
    std::unique_ptr<OCRRecognizer> CreateOCR(const std::string& name);
    
    void ListAvailableAlgorithms();
    
private:
    AlgorithmPluginManager() = default;
    std::unordered_map<std::string, std::function<std::unique_ptr<ObjectDetector>()>> detectors_;
    std::unordered_map<std::string, std::function<std::unique_ptr<ImageSegmenter>()>> segmenters_;
    std::unordered_map<std::string, std::function<std::unique_ptr<OCRRecognizer>()>> ocrs_;
};

} // namespace vision_engine

#endif // __cplusplus

#endif // VE_ALGORITHMS_H
