#include "vision_engine.h"
#include "core/ve_options.h"
#include "core/ve_logger.h"
#include "inference/ve_inference.h"
#include "quantization/ve_quantization.h"
#include "ota/ve_ota.h"
#include <memory>

namespace vision_engine {

// 全局引擎实例
static InferenceEngine* g_engine = nullptr;

void* ve_create_engine(const VeEngineOptions* options) {
    if (!options) {
        return nullptr;
    }
    
    // 初始化日志
    VeLogger::Initialize("VisionEngine", true, false);
    
    auto engine = new InferenceEngine();
    VeStatusCode status = engine->Initialize(*options);
    
    if (status != VE_SUCCESS) {
        delete engine;
        VE_LOG_ERROR("Failed to initialize engine: " + std::string(ve_status_string(status)));
        return nullptr;
    }
    
    g_engine = engine;
    return static_cast<void*>(engine);
}

void ve_destroy_engine(void* engine) {
    if (engine) {
        auto eng = static_cast<InferenceEngine*>(engine);
        delete eng;
        if (g_engine == eng) {
            g_engine = nullptr;
        }
    }
}

const char* ve_get_version() {
    return "VisionEngine v1.0.0";
}

const char* ve_get_last_error(void* engine) {
    if (engine) {
        auto eng = static_cast<InferenceEngine*>(engine);
        return eng->GetLastError().c_str();
    }
    return "Invalid engine handle";
}

} // extern "C"
