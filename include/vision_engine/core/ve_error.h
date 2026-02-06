#ifndef VE_ERROR_H
#define VE_ERROR_H

/**
 * @brief VisionEngine 错误码定义
 */

// 错误码定义
typedef enum {
    VE_SUCCESS = 0,                    // 成功
    VE_ERROR_GENERIC = -1,            // 通用错误
    VE_ERROR_INVALID_ARG = -2,        // 无效参数
    VE_ERROR_NOT_INITIALIZED = -3,    // 未初始化
    VE_ERROR_ALREADY_INITIALIZED = -4,// 已初始化
    VE_ERROR_OUT_OF_MEMORY = -5,      // 内存不足
    VE_ERROR_FILE_NOT_FOUND = -6,     // 文件不存在
    VE_ERROR_FILE_READ = -7,          // 文件读取错误
    VE_ERROR_FILE_WRITE = -8,         // 文件写入错误
    VE_ERROR_INVALID_MODEL = -9,      // 无效模型
    VE_ERROR_MODEL_LOAD_FAILED = -10, // 模型加载失败
    VE_ERROR_INFERENCE_FAILED = -11,  // 推理失败
    VE_ERROR_BACKEND_NOT_FOUND = -12, // 后端未找到
    VE_ERROR_BACKEND_INIT_FAILED = -13,// 后端初始化失败
    VE_ERROR_DEVICE_NOT_SUPPORTED = -14,// 设备不支持
    VE_ERROR_QUANTIZATION_FAILED = -15,// 量化失败
    VE_ERROR_OTA_UPDATE_FAILED = -16, // OTA更新失败
    VE_ERROR_NETWORK = -17,           // 网络错误
    VE_ERROR_TIMEOUT = -18,            // 超时
    VE_ERROR_PERMISSION_DENIED = -19, // 权限拒绝
    VE_ERROR_SIGNATURE_INVALID = -20, // 签名无效
    VE_ERROR_CHECKSUM_MISMATCH = -21, // 校验和不匹配
    VE_ERROR_VERSION_MISMATCH = -22,  // 版本不匹配
    VE_ERROR_CANCELLED = -23,         // 操作取消
    VE_ERROR_NOT_IMPLEMENTED = -24,   // 未实现
    VE_ERROR_BUFFER_TOO_SMALL = -25,  // 缓冲区太小
    VE_ERROR_THREAD_POOL_FULL = -26,  // 线程池已满
    VE_ERROR_GPU_OUT_OF_MEMORY = -27  // GPU内存不足
} VeStatusCode;

// 日志级别
typedef enum {
    VE_LOG_DEBUG = 0,
    VE_LOG_INFO,
    VE_LOG_WARN,
    VE_LOG_ERROR,
    VE_LOG_FATAL
} VeLogLevel;

// 获取错误描述
static inline const char* ve_status_string(VeStatusCode code) {
    switch (code) {
        case VE_SUCCESS:
            return "Success";
        case VE_ERROR_GENERIC:
            return "Generic error";
        case VE_ERROR_INVALID_ARG:
            return "Invalid argument";
        case VE_ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case VE_ERROR_ALREADY_INITIALIZED:
            return "Already initialized";
        case VE_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case VE_ERROR_FILE_NOT_FOUND:
            return "File not found";
        case VE_ERROR_FILE_READ:
            return "File read error";
        case VE_ERROR_FILE_WRITE:
            return "File write error";
        case VE_ERROR_INVALID_MODEL:
            return "Invalid model";
        case VE_ERROR_MODEL_LOAD_FAILED:
            return "Model load failed";
        case VE_ERROR_INFERENCE_FAILED:
            return "Inference failed";
        case VE_ERROR_BACKEND_NOT_FOUND:
            return "Backend not found";
        case VE_ERROR_BACKEND_INIT_FAILED:
            return "Backend initialization failed";
        case VE_ERROR_DEVICE_NOT_SUPPORTED:
            return "Device not supported";
        case VE_ERROR_QUANTIZATION_FAILED:
            return "Quantization failed";
        case VE_ERROR_OTA_UPDATE_FAILED:
            return "OTA update failed";
        case VE_ERROR_NETWORK:
            return "Network error";
        case VE_ERROR_TIMEOUT:
            return "Timeout";
        case VE_ERROR_PERMISSION_DENIED:
            return "Permission denied";
        case VE_ERROR_SIGNATURE_INVALID:
            return "Invalid signature";
        case VE_ERROR_CHECKSUM_MISMATCH:
            return "Checksum mismatch";
        case VE_ERROR_VERSION_MISMATCH:
            return "Version mismatch";
        case VE_ERROR_CANCELLED:
            return "Operation cancelled";
        case VE_ERROR_NOT_IMPLEMENTED:
            return "Not implemented";
        case VE_ERROR_BUFFER_TOO_SMALL:
            return "Buffer too small";
        case VE_ERROR_THREAD_POOL_FULL:
            return "Thread pool is full";
        case VE_ERROR_GPU_OUT_OF_MEMORY:
            return "GPU out of memory";
        default:
            return "Unknown error";
    }
}

// 错误处理宏
#define VE_CHECK_STATUS(status) \
    do { \
        VeStatusCode _status = (status); \
        if (_status != VE_SUCCESS) { \
            return _status; \
        } \
    } while(0)

#define VE_CHECK_NULL(ptr) \
    do { \
        if ((ptr) == nullptr) { \
            return VE_ERROR_INVALID_ARG; \
        } \
    } while(0)

#endif // VE_ERROR_H
