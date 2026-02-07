#ifndef VE_TYPES_H
#define VE_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include <string>
#include <vector>
#include <memory>
#endif

/**
 * @brief VisionEngine 基本类型定义
 */

// 句柄类型
typedef void* VeEngineHandle;
typedef void* VeModelHandle;
typedef void* VeInferenceHandle;

// 浮点类型别名
typedef float ve_float32_t;
typedef double ve_float64_t;

// 整数类型别名
typedef int8_t ve_int8_t;
typedef int16_t ve_int16_t;
typedef int32_t ve_int32_t;
typedef int64_t ve_int64_t;
typedef uint8_t ve_uint8_t;
typedef uint16_t ve_uint16_t;
typedef uint32_t ve_uint32_t;
typedef uint64_t ve_uint64_t;

// 设备类型
typedef enum {
    VE_DEVICE_CPU = 0,      // CPU设备
    VE_DEVICE_GPU,         // GPU设备 (通用)
    VE_DEVICE_CUDA,        // NVIDIA CUDA
    VE_DEVICE_VULKAN,      // Vulkan
    VE_DEVICE_OPENCL,      // OpenCL
    VE_DEVICE_METAL,       // Apple Metal
    VE_DEVICE_HEXAGON,     // Qualcomm Hexagon DSP
    VE_DEVICE_NPU          // 神经网络处理单元
} VeDeviceType;

// 数据精度
typedef enum {
    VE_PRECISION_FP32 = 0, // 32位浮点
    VE_PRECISION_FP16,     // 16位浮点
    VE_PRECISION_BF16,     // Brain Float16
    VE_PRECISION_INT8,     // 8位整数 (量化)
    VE_PRECISION_INT4,     // 4位整数
    VE_PRECISION_UINT8     // 8位无符号整数
} VePrecision;

// 推理后端
typedef enum {
    VE_BACKEND_ONNX = 0,    // ONNX Runtime
    VE_BACKEND_TENSORRT,   // NVIDIA TensorRT
    VE_BACKEND_OPENVINO,   // Intel OpenVINO
    VE_BACKEND_NCNN,       // 腾讯NCNN
    VE_BACKEND_MNN,        // 阿里MNN
    VE_BACKEND_TFLITE,     // TensorFlow Lite
    VE_BACKEND_COREML,     // Apple Core ML
    VE_BACKEND_RKNN,       // 瑞芯微RKNN
    VE_BACKEND_CUSTOM      // 自定义后端
} VeBackendType;

// 模型类型
typedef enum {
    VE_MODEL_CLASSIFICATION = 0,  // 图像分类
    VE_MODEL_DETECTION,           // 目标检测
    VE_MODEL_SEGMENTATION,        // 图像分割
    VE_MODEL_OCR,                 // OCR识别
    VE_MODEL_POSE,                // 姿态估计
    VE_MODEL_FACE,                // 人脸相关
    VE_MODEL_SR,                  // 超分辨率
    VE_MODEL_GENERAL              // 通用模型
} VeModelType;

// 任务类型
typedef enum {
    VE_TASK_IMAGE_CLASSIFICATION = 0,
    VE_TASK_OBJECT_DETECTION,
    VE_TASK_INSTANCE_SEGMENTATION,
    VE_TASK_SEMANTIC_SEGMENTATION,
    VE_TASK_POSE_ESTIMATION,
    VE_TASK_FACE_DETECTION,
    VE_TASK_FACE_RECOGNITION,
    VE_TASK_OCR_TEXT_DETECTION,
    VE_TASK_OCR_TEXT_RECOGNITION,
    VE_TASK_SUPER_RESOLUTION,
    VE_TASK_IMAGE_ENHANCEMENT,
    VE_TASK_GENERAL
} VeTaskType;

// 张量形状
typedef struct {
    int32_t* dims;      // 维度数组
    int32_t ndim;       // 维度数量
} VeTensorShape;

// 张量描述
typedef struct {
    VePrecision precision;    // 数据精度
    VeTensorShape shape;      // 形状
    const char* name;         // 名称
    bool normalized;          // 是否归一化
    float scale;              // 量化 scale
    int32_t zero_point;       // 量化 zero point
} VeTensorDesc;

// 图像格式
typedef enum {
    VE_IMAGE_FORMAT_RGB = 0,
    VE_IMAGE_FORMAT_BGR,
    VE_IMAGE_FORMAT_RGBA,
    VE_IMAGE_FORMAT_GRAY,
    VE_IMAGE_FORMAT_YUV420,
    VE_IMAGE_FORMAT_NV12,
    VE_IMAGE_FORMAT_NV21
} VeImageFormat;

// 图像数据
typedef struct {
    uint8_t* data;            // 图像数据指针
    int32_t width;           // 宽度
    int32_t height;           // 高度
    VeImageFormat format;     // 图像格式
    float* mean;              // 均值 (3通道)
    float* std;               // 标准差 (3通道)
} VeImageData;

// 批处理图像
typedef struct {
    VeImageData** images;      // 图像数组
    int32_t batch_size;       // 批大小
} VeImageBatch;

// 矩形框
typedef struct {
    float x1, y1;             // 左上角
    float x2, y2;             // 右下角
} VeBBox;

// 关键点
typedef struct {
    float x, y;               // 坐标
    float score;              // 置信度
} VeKeyPoint;

// 检测结果
typedef struct {
    VeBBox bbox;              // 边界框
    float score;              // 检测置信度
    int32_t class_id;         // 类别ID
    const char* class_name;   // 类别名称
    VeKeyPoint* keypoints;    // 关键点 (可选)
    int32_t num_keypoints;    // 关键点数量
    float* mask;              // 分割掩码 (可选)
    int32_t mask_width;       // 掩码宽度
    int32_t mask_height;      // 掩码高度
} VeDetection;

// 推理输入
typedef struct {
    VeImageBatch* images;              // 输入图像
    const char* input_name;            // 输入节点名称
    void* tensor_data;                 // 或直接的张量数据
    VeTensorDesc* tensor_desc;         // 张量描述
} VeInferenceInput;

// 推理输出
typedef struct {
    void* tensor_data;                 // 输出数据
    VeTensorDesc* tensor_desc;         // 张量描述
    int32_t batch_size;               // 批大小
} VeInferenceOutput;

// 推理请求
typedef struct {
    VeInferenceInput** inputs;         // 输入数组
    int32_t num_inputs;               // 输入数量
    VeInferenceOutput** outputs;      // 输出数组
    int32_t num_outputs;              // 输出数量
    int32_t batch_size;               // 批大小
} VeInferenceRequest;

// 推理响应
typedef struct {
    VeDetection* detections;          // 检测结果数组
    int32_t num_detections;           // 检测数量
    float inference_time_ms;          // 推理耗时 (毫秒)
    void* raw_output;                 // 原始输出
    int32_t raw_output_size;          // 原始输出大小
} VeInferenceResult;

// 性能指标
typedef struct {
    double preprocess_time_ms;         // 预处理耗时
    double inference_time_ms;          // 推理耗时
    double postprocess_time_ms;        // 后处理耗时
    double total_time_ms;              // 总耗时
    size_t memory_used_bytes;          // 内存使用量
    float gpu_memory_used_mb;          // GPU内存使用
} VePerformanceMetrics;

// 模型信息
typedef struct {
    const char* name;                  // 模型名称
    const char* path;                  // 模型路径
    VeBackendType backend;             // 推理后端
    VePrecision precision;            // 数据精度
    VeModelType model_type;            // 模型类型
    int32_t input_width;               // 输入宽度
    int32_t input_height;              // 输入高度
    int32_t input_channels;           // 输入通道数
    int32_t num_classes;               // 类别数量
    float confidence_threshold;        // 置信度阈值
    float nms_threshold;              // NMS阈值
    const char* version;               // 模型版本
} VeModelInfo;

// 引擎配置
typedef struct {
    VeBackendType preferred_backend;   // 首选后端
    VeDeviceType device_type;         // 设备类型
    VePrecision precision;            // 推理精度
    int32_t num_threads;               // 线程数
    bool enable_profiling;            // 启用性能分析
    bool enable_debug;                // 启用调试
    const char* model_path;           // 模型路径
    const char* config_path;          // 配置文件路径
    const char* cache_path;          // 缓存路径
    int32_t batch_size;               // 默认批大小
    int32_t gpu_id;                   // GPU设备ID
} VeEngineOptions;

// 量化配置 (C API)
typedef struct {
    int32_t type;                     // 量化类型
    int32_t calibration_method;        // 校准方法
    int32_t calibration_samples;       // 校准样本数
    bool mixed_precision;              // 混合精度
    const char* fp16_layers;          // FP16层 (逗号分隔)
} VeQuantConfig;

// 量化评估指标 (C API)
typedef struct {
    float accuracy_drop;              // 精度损失
    float inference_speedup;          // 推理加速比
    size_t model_size_reduction;      // 模型大小减少量
    float compression_ratio;          // 压缩比
} VeQuantizationMetrics;

// 回调函数类型
typedef void (*VeProgressCallback)(float progress, const char* message);
typedef void (*VeLogCallback)(int level, const char* message);
typedef void (*VeInferenceCallback)(const VeInferenceResult* result);

#ifdef __cplusplus

// C++ 封装类型
using VeString = std::string;
using VeTensor = std::vector<float>;
using VeDetections = std::vector<VeDetection>;
using VeTensorShapeVec = std::vector<int64_t>;

#endif

#endif // VE_TYPES_H
