#ifndef VISION_ENGINE_H
#define VISION_ENGINE_H

/**
 * @brief VisionEngine 核心头文件
 * 
 * 跨平台视觉推理引擎
 * 支持多种推理后端: ONNX Runtime, TensorRT, NCNN
 * 支持INT8量化、OTA热更新
 */

#include "core/ve_types.h"
#include "core/ve_error.h"
#include "core/ve_options.h"
#include "inference/ve_inference.h"
#include "inference/ve_model.h"
#include "inference/ve_result.h"
#include "quantization/ve_quantization.h"
#include "ota/ve_ota.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建VisionEngine实例
 * @param options 引擎配置选项
 * @return 引擎句柄，失败返回nullptr
 */
VE_API void* ve_create_engine(const VeEngineOptions* options);

/**
 * @brief 销毁VisionEngine实例
 * @param engine 引擎句柄
 */
VE_API void ve_destroy_engine(void* engine);

/**
 * @brief 获取引擎版本信息
 * @return 版本字符串
 */
VE_API const char* ve_get_version();

/**
 * @brief 获取最后错误信息
 * @param engine 引擎句柄
 * @return 错误描述字符串
 */
VE_API const char* ve_get_last_error(void* engine);

#ifdef __cplusplus
}
#endif

#endif // VISION_ENGINE_H
