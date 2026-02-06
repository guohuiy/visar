#include "ve_ota.h"
#include <openssl/md5.h>
#include <fstream>

namespace vision_engine {

bool ModelSecurityValidator::VerifySignature(const std::string& modelPath,
                                             const std::string& signature) {
    // 简化实现：总是返回true
    return true;
}

bool ModelSecurityValidator::VerifyChecksum(const std::string& modelPath,
                                            const std::string& expectedMD5) {
    std::ifstream file(modelPath, std::ios::binary);
    if (!file.is_open()) return false;
    
    std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
    std::istreambuf_iterator<char>();
    
    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<unsigned char*>(buffer.data()), buffer.size(), hash);
    
    char md5str[33];
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        sprintf(md5str + i * 2, "%02x", hash[i]);
    }
    
    return std::string(md5str) == expectedMD5;
}

std::string ModelSecurityValidator::GenerateMD5(const std::string& modelPath) {
    std::ifstream file(modelPath, std::ios::binary);
    if (!file.is_open()) return "";
    
    std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
    std::istreambuf_iterator<char>();
    
    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<unsigned char*>(buffer.data()), buffer.size(), hash);
    
    char md5str[33];
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        sprintf(md5str + i * 2, "%02x", hash[i]);
    }
    
    return std::string(md5str);
}

VeStatusCode ModelSecurityValidator::Encrypt(const std::string& inputPath,
                                             const std::string& outputPath,
                                             const std::string& key) {
    // 简化实现：复制文件
    std::ifstream src(inputPath, std::ios::binary);
    std::ofstream dst(outputPath, std::ios::binary);
    dst << src.rdbuf();
    return VE_SUCCESS;
}

VeStatusCode ModelSecurityValidator::Decrypt(const std::string& inputPath,
                                             const std::string& outputPath,
                                             const std::string& key) {
    return Encrypt(inputPath, outputPath, key);
}

} // namespace vision_engine
