#ifndef VE_LOGGER_H
#define VE_LOGGER_H

#include "ve_error.h"
#include <string>

#ifdef __cplusplus
namespace vision_engine {

class VeLogger {
public:
    static void Initialize(const std::string& name = "VisionEngine",
                          bool console = true, bool file = false);
    static void SetLevel(VeLogLevel level);
    static void Debug(const std::string& message);
    static void Info(const std::string& message);
    static void Warn(const std::string& message);
    static void Error(const std::string& message);
    static void Fatal(const std::string& message);
    static void Shutdown();
private:
    static void Log(VeLogLevel level, const std::string& message);
};

} // namespace vision_engine

#define VE_LOG_DEBUG(msg) vision_engine::VeLogger::Debug(msg)
#define VE_LOG_INFO(msg)  vision_engine::VeLogger::Info(msg)
#define VE_LOG_WARN(msg)  vision_engine::VeLogger::Warn(msg)
#define VE_LOG_ERROR(msg) vision_engine::VeLogger::Error(msg)
#define VE_LOG_FATAL(msg) vision_engine::VeLogger::Fatal(msg)

#endif // __cplusplus

#endif // VE_LOGGER_H
