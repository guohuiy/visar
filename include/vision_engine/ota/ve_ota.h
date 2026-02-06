#ifndef VE_OTA_H
#define VE_OTA_H

#include "../core/ve_types.h"
#include "../core/ve_error.h"

// VE_API 宏定义 - 用于导出/导入符号
#ifdef _WIN32
    #ifdef VISION_ENGINE_EXPORT
        #define VE_API __declspec(dllexport)
    #else
        #define VE_API __declspec(dllimport)
    #endif
#else
    #define VE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <chrono>

namespace vision_engine {

/**
 * @brief 更新策略
 */
enum class UpdateStrategy {
    IMMEDIATE = 0,    // 立即更新
    ON_IDLE,          // 空闲时更新
    BACKGROUND,       // 后台静默更新
    MANUAL            // 手动触发
};

/**
 * @brief 更新信息
 */
struct UpdateInfo {
    std::string version;           // 新版本号
    std::string downloadUrl;       // 下载地址
    size_t fileSize;              // 文件大小
    std::string md5Checksum;      // MD5校验和
    std::string releaseNotes;      // 发布说明
    bool isMandatory = false;     // 是否强制更新
    bool deltaUpdate = false;     // 是否支持增量更新
    std::string deltaVersion;     // 增量更新源版本
};

/**
 * @brief 版本信息
 */
struct VersionInfo {
    std::string version;           // 版本号
    std::string md5Checksum;      // MD5校验和
    size_t fileSize;              // 文件大小
    std::string releaseNotes;      // 发布说明
    std::chrono::system_clock::time_point releaseDate;
    bool isMandatory = false;     // 是否强制更新
};

/**
 * @brief OTA配置
 */
struct OTAConfig {
    std::string serverURL;         // 模型服务器地址
    std::string currentVersion;    // 当前模型版本
    bool autoUpdate = false;       // 是否自动更新
    UpdateStrategy strategy = UpdateStrategy::MANUAL;
    std::string cachePath;         // 本地缓存路径
    bool enableDeltaUpdate = true; // 启用增量更新
    std::string encryptionKey;     // 模型加密密钥
};

/**
 * @brief 模型OTA更新器
 */
class ModelOTAUpdater {
public:
    ModelOTAUpdater();
    ~ModelOTAUpdater();

    /**
     * @brief 初始化
     * @param config OTA配置
     * @return 状态码
     */
    VeStatusCode Initialize(const OTAConfig& config);

    /**
     * @brief 检查更新
     * @return 更新信息 (如果没有更新则version为空)
     */
    UpdateInfo CheckForUpdates();

    /**
     * @brief 下载新模型
     * @param info 更新信息
     * @return 状态码
     */
    VeStatusCode DownloadModel(const UpdateInfo& info);

    /**
     * @brief 增量更新下载
     * @param info 更新信息
     * @return 状态码
     */
    VeStatusCode DownloadDeltaUpdate(const UpdateInfo& info);

    /**
     * @brief 取消下载
     */
    void CancelDownload();

    /**
     * @brief 验证模型
     * @param modelPath 模型路径
     * @return 状态码
     */
    VeStatusCode ValidateModel(const std::string& modelPath);

    /**
     * @brief 原子化替换模型
     * @param newModelPath 新模型路径
     * @return 状态码
     */
    VeStatusCode SwapModel(const std::string& newModelPath);

    /**
     * @brief 回滚到上一版本
     * @return 状态码
     */
    VeStatusCode Rollback();

    /**
     * @brief 解密模型
     * @param encryptedPath 加密模型路径
     * @param outputPath 输出路径
     * @return 状态码
     */
    VeStatusCode DecryptModel(const std::string& encryptedPath,
                             std::string& outputPath);

    /**
     * @brief 加密模型
     * @param inputPath 输入模型路径
     * @param outputPath 输出路径
     * @return 状态码
     */
    VeStatusCode EncryptModel(const std::string& inputPath,
                              std::string& outputPath);

    /**
     * @brief 获取当前版本
     */
    VersionInfo GetCurrentVersion() const;

    /**
     * @brief 获取版本历史
     */
    std::vector<VersionInfo> GetVersionHistory() const;

    /**
     * @brief 获取下载进度
     */
    float GetDownloadProgress() const;

    /**
     * @brief 获取下载速度 (bytes/s)
     */
    double GetDownloadSpeed() const;

    /**
     * @brief 设置进度回调
     */
    void SetProgressCallback(std::function<void(float, const std::string&)> callback);

    /**
     * @brief 设置下载完成回调
     */
    void SetDownloadCompleteCallback(std::function<void(const UpdateInfo&)> callback);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 模型版本管理器
 */
class ModelVersionManager {
public:
    ModelVersionManager();
    ~ModelVersionManager();

    /**
     * @brief 加载版本信息
     */
    VeStatusCode Load(const std::string& versionFilePath);

    /**
     * @brief 保存版本信息
     */
    VeStatusCode Save(const std::string& versionFilePath);

    /**
     * @brief 获取当前版本
     */
    VersionInfo GetCurrentVersion() const;

    /**
     * @brief 设置当前版本
     */
    void SetCurrentVersion(const VersionInfo& version);

    /**
     * @brief 添加版本
     */
    void AddVersion(const VersionInfo& version);

    /**
     * @brief 删除版本
     */
    void RemoveVersion(const std::string& version);

    /**
     * @brief 检查是否有新版本
     */
    bool HasNewVersion(const VersionInfo& remote);

    /**
     * @brief 获取所有版本
     */
    std::vector<VersionInfo> GetAllVersions() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 模型安全验证器
 */
class ModelSecurityValidator {
public:
    /**
     * @brief 验证数字签名
     */
    bool VerifySignature(const std::string& modelPath,
                         const std::string& signature);

    /**
     * @brief 验证完整性校验
     */
    bool VerifyChecksum(const std::string& modelPath,
                        const std::string& expectedMD5);

    /**
     * @brief 生成MD5校验和
     */
    std::string GenerateMD5(const std::string& modelPath);

    /**
     * @brief 加密模型 (AES-256)
     */
    VeStatusCode Encrypt(const std::string& inputPath,
                         const std::string& outputPath,
                         const std::string& key);

    /**
     * @brief 解密模型 (AES-256)
     */
    VeStatusCode Decrypt(const std::string& inputPath,
                         const std::string& outputPath,
                         const std::string& key);
};

} // namespace vision_engine

#endif // __cplusplus

// ============================================================================
// C API 类型定义
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 更新策略枚举 (C API)
 */
typedef enum {
    VE_UPDATE_STRATEGY_IMMEDIATE = 0,
    VE_UPDATE_STRATEGY_ON_IDLE,
    VE_UPDATE_STRATEGY_BACKGROUND,
    VE_UPDATE_STRATEGY_MANUAL
} VeUpdateStrategy;

/**
 * @brief 更新信息 (C API)
 */
typedef struct {
    char version[64];              // 新版本号
    char downloadUrl[512];         // 下载地址
    size_t fileSize;              // 文件大小
    char md5Checksum[64];          // MD5校验和
    char releaseNotes[1024];       // 发布说明
    int isMandatory;              // 是否强制更新
    int deltaUpdate;               // 是否支持增量更新
    char deltaVersion[64];         // 增量更新源版本
} VeUpdateInfo;

/**
 * @brief 版本信息 (C API)
 */
typedef struct {
    char version[64];             // 版本号
    char md5Checksum[64];         // MD5校验和
    size_t fileSize;             // 文件大小
    char releaseNotes[1024];      // 发布说明
    int64_t releaseDate;          // 发布日期 (Unix时间戳)
    int isMandatory;             // 是否强制更新
} VeVersionInfo;

/**
 * @brief OTA配置 (C API)
 */
typedef struct {
    char serverURL[512];          // 模型服务器地址
    char currentVersion[64];      // 当前模型版本
    int autoUpdate;               // 是否自动更新
    VeUpdateStrategy strategy;     // 更新策略
    char cachePath[512];          // 本地缓存路径
    int enableDeltaUpdate;         // 启用增量更新
    char encryptionKey[256];       // 模型加密密钥
} VeOTAConfig;

/**
 * @brief 进度回调函数类型 (C API)
 */
typedef void (*VeOTAProgressCallback)(float progress, const char* message);

/**
 * @brief 下载完成回调函数类型 (C API)
 */
typedef void (*VeOTADownloadCompleteCallback)(const VeUpdateInfo* info);

// ============================================================================
// C API 函数声明
// ============================================================================

/**
 * @brief 创建OTA更新器
 * @param config OTA配置
 * @return 更新器句柄，失败返回nullptr
 */
VE_API void* ve_ota_create(const VeOTAConfig* config);

/**
 * @brief 销毁OTA更新器
 * @param handle 更新器句柄
 */
VE_API void ve_ota_destroy(void* handle);

/**
 * @brief 检查更新
 * @param handle 更新器句柄
 * @return 更新信息
 */
VE_API VeUpdateInfo ve_ota_check_update(void* handle);

/**
 * @brief 下载模型
 * @param handle 更新器句柄
 * @param info 更新信息
 * @param callback 进度回调 (可选，传递nullptr表示不需要)
 * @return 状态码
 */
VE_API VeStatusCode ve_ota_download(void* handle,
                                     const VeUpdateInfo* info,
                                     VeOTAProgressCallback callback);

/**
 * @brief 增量更新下载
 * @param handle 更新器句柄
 * @param info 更新信息
 * @param callback 进度回调 (可选)
 * @return 状态码
 */
VE_API VeStatusCode ve_ota_download_delta(void* handle,
                                           const VeUpdateInfo* info,
                                           VeOTAProgressCallback callback);

/**
 * @brief 取消下载
 * @param handle 更新器句柄
 */
VE_API void ve_ota_cancel_download(void* handle);

/**
 * @brief 验证模型
 * @param handle 更新器句柄
 * @param model_path 模型路径
 * @return 状态码
 */
VE_API VeStatusCode ve_ota_validate(void* handle, const char* model_path);

/**
 * @brief 替换模型
 * @param handle 更新器句柄
 * @param new_model_path 新模型路径
 * @return 状态码
 */
VE_API VeStatusCode ve_ota_swap(void* handle, const char* new_model_path);

/**
 * @brief 回滚
 * @param handle 更新器句柄
 * @return 状态码
 */
VE_API VeStatusCode ve_ota_rollback(void* handle);

/**
 * @brief 获取当前版本
 * @param handle 更新器句柄
 * @return 版本信息
 */
VE_API VeVersionInfo ve_ota_get_current_version(void* handle);

/**
 * @brief 获取下载进度
 * @param handle 更新器句柄
 * @return 进度值 (0.0 ~ 1.0)
 */
VE_API float ve_ota_get_progress(void* handle);

/**
 * @brief 获取下载速度 (bytes/s)
 * @param handle 更新器句柄
 * @return 下载速度
 */
VE_API double ve_ota_get_speed(void* handle);

/**
 * @brief 设置进度回调
 * @param handle 更新器句柄
 * @param callback 回调函数
 */
VE_API void ve_ota_set_progress_callback(void* handle, 
                                          VeOTAProgressCallback callback);

/**
 * @brief 设置下载完成回调
 * @param handle 更新器句柄
 * @param callback 回调函数
 */
VE_API void ve_ota_set_download_complete_callback(void* handle,
                                                   VeOTADownloadCompleteCallback callback);

#ifdef __cplusplus
}
#endif

#endif // VE_OTA_H
