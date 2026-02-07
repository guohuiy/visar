#include "vision_engine.h"
#include "core/ve_options.h"
#include "core/ve_logger.h"
#include <memory>

namespace vision_engine {

// 简化的引擎实现
class EngineImpl {
public:
    EngineImpl() {}
    ~EngineImpl() {}
    
    VeStatusCode Initialize(const VeEngineOptions& options) {
        VE_LOG_INFO("VisionEngine initialized");
        return VE_SUCCESS;
    }
    
    std::string GetLastError() const {
        return last_error_;
    }
    
    void SetLastError(const std::string& error) {
        last_error_ = error;
    }
    
private:
    std::string last_error_;
};

// 全局引擎实例
static EngineImpl* g_engine = nullptr;

} // namespace vision_engine

// C API实现 - 在全局命名空间，使用 extern "C"
extern "C" {

void* ve_create_engine(const VeEngineOptions* options) {
    if (!options) {
        return nullptr;
    }
    
    // 初始化日志
    VeLogger::Initialize("VisionEngine", true, false);
    
    auto engine = new vision_engine::EngineImpl();
    VeStatusCode status = engine->Initialize(*options);
    
    if (status != VE_SUCCESS) {
        delete engine;
        return nullptr;
    }
    
    vision_engine::g_engine = engine;
    return static_cast<void*>(engine);
}

void ve_destroy_engine(void* engine) {
    if (engine) {
        auto eng = static_cast<vision_engine::EngineImpl*>(engine);
        delete eng;
        if (vision_engine::g_engine == eng) {
            vision_engine::g_engine = nullptr;
        }
    }
}

const char* ve_get_version() {
    return "VisionEngine v1.0.0";
}

const char* ve_get_last_error(void* engine) {
    if (engine) {
        auto eng = static_cast<vision_engine::EngineImpl*>(engine);
        return eng->GetLastError().c_str();
    }
    return "Invalid engine handle";
}

} // extern "C"
