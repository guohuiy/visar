#include "ve_logger.h"
#include <iostream>
#include <mutex>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace vision_engine {

static std::mutex logger_mutex;
static VeLogLevel current_level_ = VE_LOG_INFO;
static bool initialized_ = false;
static std::string logger_name_ = "VisionEngine";

std::string GetCurrentTime() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

const char* LevelToString(VeLogLevel level) {
    switch (level) {
        case VE_LOG_DEBUG: return "DEBUG";
        case VE_LOG_INFO:  return "INFO";
        case VE_LOG_WARN:  return "WARN";
        case VE_LOG_ERROR: return "ERROR";
        case VE_LOG_FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

void SetConsoleColor(VeLogLevel level) {
#ifdef _WIN32
    switch (level) {
        case VE_LOG_DEBUG: SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 8); break;  // 灰色
        case VE_LOG_INFO:  SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 10); break; // 绿色
        case VE_LOG_WARN:  SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 14); break; // 黄色
        case VE_LOG_ERROR: SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 12); break; // 红色
        case VE_LOG_FATAL: SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 79); break; // 红底白字
        default: SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 7); break; // 默认白色
    }
#endif
}

void ResetConsoleColor() {
#ifdef _WIN32
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 7);
#endif
}

void VeLogger::Initialize(const std::string& name, bool console, bool file) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (initialized_) {
        return;
    }
    logger_name_ = name;
    initialized_ = true;
    (void)console;  // 控制台输出总是启用
    (void)file;     // 文件输出暂不实现
}

void VeLogger::SetLevel(VeLogLevel level) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    current_level_ = level;
}

void VeLogger::Debug(const std::string& message) {
    Log(VE_LOG_DEBUG, message);
}

void VeLogger::Info(const std::string& message) {
    Log(VE_LOG_INFO, message);
}

void VeLogger::Warn(const std::string& message) {
    Log(VE_LOG_WARN, message);
}

void VeLogger::Error(const std::string& message) {
    Log(VE_LOG_ERROR, message);
}

void VeLogger::Fatal(const std::string& message) {
    Log(VE_LOG_FATAL, message);
}

void VeLogger::Log(VeLogLevel level, const std::string& message) {
    if (level < current_level_) return;
    
    std::lock_guard<std::mutex> lock(logger_mutex);
    
    SetConsoleColor(level);
    std::cout << "[" << GetCurrentTime() << "] [" << LevelToString(level) << "] [" 
              << logger_name_ << "] " << message << std::endl;
    ResetConsoleColor();
}

void VeLogger::Shutdown() {
    std::lock_guard<std::mutex> lock(logger_mutex);
    initialized_ = false;
}

} // namespace vision_engine
