#include "ve_ota.h"
#include <curl/curl.h>

namespace vision_engine {

// 简化实现
ModelVersionManager::ModelVersionManager() = default;
ModelVersionManager::~ModelVersionManager() = default;

VeStatusCode ModelVersionManager::Load(const std::string& versionFilePath) {
    return VE_SUCCESS;
}

VeStatusCode ModelVersionManager::Save(const std::string& versionFilePath) {
    return VE_SUCCESS;
}

VersionInfo ModelVersionManager::GetCurrentVersion() const {
    return VersionInfo{};
}

void ModelVersionManager::SetCurrentVersion(const VersionInfo& version) {}

void ModelVersionManager::AddVersion(const VersionInfo& version) {}

void ModelVersionManager::RemoveVersion(const std::string& version) {}

bool ModelVersionManager::HasNewVersion(const VersionInfo& remote) {
    return false;
}

std::vector<VersionInfo> ModelVersionManager::GetAllVersions() const {
    return {};
}

} // namespace vision_engine
