#include "ve_model.h"
#include <fstream>
#include <sstream>

namespace vision_engine {

class ModelLoader::Impl {
public:
    std::string model_path_;
    VeModelInfo model_info_;
    bool loaded_ = false;
};

ModelLoader::ModelLoader() : impl_(std::make_unique<Impl>()) {}

ModelLoader::~ModelLoader() {
    Unload();
}

VeStatusCode ModelLoader::LoadModel(const std::string& model_path) {
    // 检查文件是否存在
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        return VE_ERROR_FILE_NOT_FOUND;
    }
    
    impl_->model_path_ = model_path;
    impl_->model_info_.path = impl_->model_path_.c_str();
    impl_->model_info_.name = "Unknown Model";
    impl_->loaded_ = true;
    
    return VE_SUCCESS;
}

VeStatusCode ModelLoader::LoadModelFromMemory(const void* data, size_t size) {
    if (!data || size == 0) {
        return VE_ERROR_INVALID_ARG;
    }
    
    impl_->model_path_ = "";
    impl_->loaded_ = true;
    
    return VE_SUCCESS;
}

void ModelLoader::Unload() {
    impl_->loaded_ = false;
    impl_->model_path_.clear();
}

VeModelInfo ModelLoader::GetModelInfo() const {
    return impl_->model_info_;
}

std::vector<std::string> ModelLoader::GetInputNames() const {
    return {"input"};
}

std::vector<std::string> ModelLoader::GetOutputNames() const {
    return {"output"};
}

class ModelManager::Impl {
public:
    std::unordered_map<std::string, std::shared_ptr<ModelLoader>> models_;
    std::string model_path_;
};

ModelManager::ModelManager() : impl_(std::make_unique<Impl>()) {}

ModelManager::~ModelManager() = default;

VeStatusCode ModelManager::RegisterModel(const std::string& name, const std::string& path) {
    auto loader = std::make_shared<ModelLoader>();
    VeStatusCode status = loader->LoadModel(path);
    if (status != VE_SUCCESS) {
        return status;
    }
    impl_->models_[name] = loader;
    return VE_SUCCESS;
}

void ModelManager::UnregisterModel(const std::string& name) {
    impl_->models_.erase(name);
}

std::shared_ptr<ModelLoader> ModelManager::GetModel(const std::string& name) {
    auto it = impl_->models_.find(name);
    if (it != impl_->models_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::string> ModelManager::ListModels() const {
    std::vector<std::string> names;
    for (const auto& pair : impl_->models_) {
        names.push_back(pair.first);
    }
    return names;
}

void ModelManager::SetModelPath(const std::string& path) {
    impl_->model_path_ = path;
}

std::string ModelManager::GetModelPath() const {
    return impl_->model_path_;
}

// C API实现
VeModelHandle ve_model_loader_create() {
    return static_cast<VeModelHandle>(new ModelLoader());
}

void ve_model_loader_destroy(VeModelHandle handle) {
    if (handle) {
        delete static_cast<ModelLoader*>(handle);
    }
}

VeStatusCode ve_model_load(VeModelHandle handle, const char* model_path) {
    if (!handle || !model_path) {
        return VE_ERROR_INVALID_ARG;
    }
    auto loader = static_cast<ModelLoader*>(handle);
    return loader->LoadModel(model_path);
}

VeStatusCode ve_model_load_from_memory(VeModelHandle handle, const void* data, size_t size) {
    if (!handle || !data) {
        return VE_ERROR_INVALID_ARG;
    }
    auto loader = static_cast<ModelLoader*>(handle);
    return loader->LoadModelFromMemory(data, size);
}

void ve_model_unload(VeModelHandle handle) {
    if (handle) {
        auto loader = static_cast<ModelLoader*>(handle);
        loader->Unload();
    }
}

VeModelInfo ve_model_get_info(VeModelHandle handle) {
    if (handle) {
        auto loader = static_cast<ModelLoader*>(handle);
        return loader->GetModelInfo();
    }
    return {};
}

} // namespace vision_engine
