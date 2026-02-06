#ifndef VE_MODEL_H
#define VE_MODEL_H

#include "ve_types.h"
#include "ve_error.h"

#ifdef __cplusplus
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace vision_engine {

/**
 * @brief 模型加载器
 */
class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    /**
     * @brief 加载模型文件
     * @param model_path 模型文件路径
     * @return 状态码
     */
    VeStatusCode LoadModel(const std::string& model_path);

    /**
     * @brief 加载模型数据
     * @param model_data 模型数据指针
     * @param data_size 数据大小
     * @return 状态码
     */
    VeStatusCode LoadModelFromMemory(const void* model_data, size_t data_size);

    /**
     * @brief 卸载模型
     */
    void Unload();

    /**
     * @brief 获取模型信息
     * @return 模型信息
     */
    VeModelInfo GetModelInfo() const;

    /**
     * @brief 获取输入节点名称
     * @return 输入节点名称列表
     */
    std::vector<std::string> GetInputNames() const;

    /**
     * @brief 获取输出节点名称
     * @return 输出节点名称列表
     */
    std::vector<std::string> GetOutputNames() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 模型管理器
 */
class ModelManager {
public:
    ModelManager();
    ~ModelManager();

    /**
     * @brief 注册模型
     * @param name 模型名称
     * @param path 模型路径
     * @return 状态码
     */
    VeStatusCode RegisterModel(const std::string& name, const std::string& path);

    /**
     * @brief 注销模型
     * @param name 模型名称
     */
    void UnregisterModel(const std::string& name);

    /**
     * @brief 获取模型
     * @param name 模型名称
     * @return 模型指针
     */
    std::shared_ptr<ModelLoader> GetModel(const std::string& name);

    /**
     * @brief 获取所有已注册模型
     * @return 模型名称列表
     */
    std::vector<std::string> ListModels() const;

    /**
     * @brief 设置模型路径
     * @param path 模型目录路径
     */
    void SetModelPath(const std::string& path);

    /**
     * @brief 获取模型路径
     * @return 模型目录路径
     */
    std::string GetModelPath() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_engine

#endif // __cplusplus

// C API
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建模型加载器
 */
VE_API VeModelHandle ve_model_loader_create();

/**
 * @brief 销毁模型加载器
 */
VE_API void ve_model_loader_destroy(VeModelHandle handle);

/**
 * @brief 加载模型文件
 */
VE_API VeStatusCode ve_model_load(VeModelHandle handle, const char* model_path);

/**
 * @brief 从内存加载模型
 */
VE_API VeStatusCode ve_model_load_from_memory(VeModelHandle handle, 
                                               const void* data, 
                                               size_t size);

/**
 * @brief 卸载模型
 */
VE_API void ve_model_unload(VeModelHandle handle);

/**
 * @brief 获取模型信息
 */
VE_API VeModelInfo ve_model_get_info(VeModelHandle handle);

#ifdef __cplusplus
}
#endif

#endif // VE_MODEL_H
