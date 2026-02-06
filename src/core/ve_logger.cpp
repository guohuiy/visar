#include "ve_logger.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <mutex>

namespace vision_engine {

static std::mutex logger_mutex;
static std::shared_ptr<spdlog::logger> logger_;
static bool initialized_ = false;

void VeLogger::Initialize(const std::string& name, bool console, bool file) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    
    if (initialized_) {
        return;
    }
    
    std::vector<spdlog::sink_ptr> sinks;
    
    if (console) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::debug);
        sinks.push_back(console_sink);
    }
    
    if (file) {
        try {
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("vision_engine.log", true);
            file_sink->set_level(spdlog::level::debug);
            sinks.push_back(file_sink);
        } catch (...) {
            // 忽略文件创建失败
        }
    }
    
    logger_ = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
    spdlog::register_logger(logger_);
    logger_->set_level(spdlog::level::debug);
    logger_->flush_on(spdlog::level::err);
    initialized_ = true;
}

void VeLogger::SetLevel(VeLogLevel level) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (logger_) {
        spdlog::level::level_enum spd_level;
        switch (level) {
            case VE_LOG_DEBUG: spd_level = spdlog::level::debug; break;
            case VE_LOG_INFO:  spd_level = spdlog::level::info; break;
            case VE_LOG_WARN:  spd_level = spdlog::level::warn; break;
            case VE_LOG_ERROR: spd_level = spdlog::level::err; break;
            case VE_LOG_FATAL: spd_level = spdlog::level::critical; break;
            default: spd_level = spdlog::level::info; break;
        }
        logger_->set_level(spd_level);
    }
}

void VeLogger::Debug(const std::string& message) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (logger_) logger_->debug(message);
}

void VeLogger::Info(const std::string& message) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (logger_) logger_->info(message);
}

void VeLogger::Warn(const std::string& message) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (logger_) logger_->warn(message);
}

void VeLogger::Error(const std::string& message) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (logger_) logger_->error(message);
}

void VeLogger::Fatal(const std::string& message) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (logger_) logger_->critical(message);
}

void VeLogger::Shutdown() {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (logger_) {
        logger_->flush();
        spdlog::drop_all();
        logger_.reset();
        initialized_ = false;
    }
}

} // namespace vision_engine
