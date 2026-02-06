#ifndef VE_LOGGER_H
#define VE_LOGGER_H

#include "ve_error.h"
#include <string>

#ifdef __cplusplus
namespace vision_engine {

/**
 * @brief 日志管理器
 */
class VeLogger {
public:
    /**
     * @brief 初始化日志系统
     * @param name logger名称
     * @param console 是否输出到控制台
     * @param file 是否输出到文件
     */
    static void Initialize(const std::string& name = "VisionEngine",
                          bool console = true,
                          bool file = false);

    /**
     * @brief 设置日志级别
     */
    static void SetLevel(VeLogLevel level);

    /**
     * @brief Debug日志
     */
    static void Debug(const std::string& message);

    /**
     * @brief Info日志
     */
    static void Info(const std::string& message);

    /**
     * @brief Warning日志
     */
    static void Warn(const std::string& message);

    /**
     * @brief Error日志
     */
    static void Error(const std::string& message);

    /**
     * @brief Fatal日志
     */
    static void Fatal(const std::string& message);

    /**
     * @brief 关闭日志系统
     */
    static void Shutdown();
};

} // namespace vision_engine

// 便捷宏
#define VE_LOG_DEBUG(msg) vision_engine::VeLogger::Debug(msg)
#define VE_LOG_INFO(msg)  vision_engine::VeLogger::Info(msg)
#define VE_LOG_WARN(msg)  vision_engine::VeLogger::Warn(msg)
#define VE_LOG_ERROR(msg) vision_engine::VeLogger::Error(msg)
#define VE_LOG_FATAL(msg) vision_engine::VeLogger::Fatal(msg)

#endif // __cplusplus

#endif // VE_LOGGER_H
