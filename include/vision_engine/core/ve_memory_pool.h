/**
 * @file ve_memory_pool.h
 * @brief 内存池管理器 - 减少动态内存分配开销
 * @author VisionEngine Team
 * @date 2024-02
 */

#ifndef VE_MEMORY_POOL_H
#define VE_MEMORY_POOL_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <map>
#include <algorithm>

#ifdef _WIN32
    #include <malloc.h>
    #define VE_ALIGNED_ALLOC(align, size) _aligned_malloc(size, align)
    #define VE_ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #include <stdlib.h>
    #define VE_ALIGNED_ALLOC(align, size) aligned_alloc(align, size)
    #define VE_ALIGNED_FREE(ptr) free(ptr)
#endif

namespace vision_engine {

/**
 * @brief 内存池配置
 */
struct MemoryPoolConfig {
    size_t max_pool_size_mb = 256;      // 最大池内存 (MB)
    size_t default_block_size = 1024 * 1024;  // 默认块大小 1MB
    size_t max_block_size = 16 * 1024 * 1024; // 最大块大小 16MB
    size_t alignment = 64;              // 内存对齐 (AVX2需要64字节)
};

/**
 * @brief 张量内存池
 * 
 * 特性：
 * - 预分配内存池，减少malloc/free开销
 * - 64字节对齐，支持SIMD优化
 * - 线程安全
 */
class TensorMemoryPool {
public:
    static TensorMemoryPool& Instance() {
        static TensorMemoryPool instance;
        return instance;
    }
    
    /**
     * @brief 从池中获取内存
     * @param size 需要的内存大小 (字节)
     * @return 预分配的float数组指针
     */
    float* Allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t num_elements = (size + sizeof(float) - 1) / sizeof(float);
        
        // 查找合适大小的缓存块
        auto it = free_blocks_.lower_bound(num_elements);
        if (it != free_blocks_.end()) {
            float* ptr = it->second;
            free_blocks_.erase(it);
            used_[ptr] = num_elements;
            return ptr;
        }
        
        // 创建新块
        size_t new_size = std::max(num_elements, config_.default_block_size / sizeof(float));
        new_size = ((new_size + 63) / 64) * 64;  // 64字节对齐
        
        float* ptr = static_cast<float*>(VE_ALIGNED_ALLOC(
            config_.alignment, 
            new_size * sizeof(float)
        ));
        
        if (!ptr) {
            return nullptr;
        }
        
        used_[ptr] = new_size;
        return ptr;
    }
    
    /**
     * @brief 分配指定数量的float元素
     */
    float* AllocateTensor(int width, int height, int channels = 3) {
        return Allocate(static_cast<size_t>(width) * height * channels);
    }
    
    /**
     * @brief 将内存归还到池中
     */
    void Deallocate(float* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = used_.find(ptr);
        if (it != used_.end()) {
            size_t size = it->second;
            used_.erase(it);
            
            // 检查是否超过池大小限制
            size_t total_free = 0;
            for (const auto& block : free_blocks_) {
                total_free += block.first * block.second;
            }
            
            if (total_free < config_.max_pool_size_mb * 1024 * 1024 / sizeof(float)) {
                free_blocks_[size] = ptr;
            } else {
            // 超过限制，释放内存
            VE_ALIGNED_FREE(ptr);
            }
        }
    }
    
    /**
     * @brief 清理池中所有缓存
     */
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& block : free_blocks_) {
            VE_ALIGNED_FREE(block.second);
        }
        free_blocks_.clear();
        used_.clear();
    }
    
    /**
     * @brief 获取当前池状态
     */
    struct PoolStats {
        size_t used_count;
        size_t free_count;
        size_t used_memory_bytes;
        size_t free_memory_bytes;
    };
    
    PoolStats GetStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        PoolStats stats;
        stats.used_count = used_.size();
        stats.free_count = free_blocks_.size();
        stats.used_memory_bytes = 0;
        stats.free_memory_bytes = 0;
        
        for (const auto& p : used_) {
            stats.used_memory_bytes += p.second * sizeof(float);
        }
        for (const auto& p : free_blocks_) {
            stats.free_memory_bytes += p.first * p.second * sizeof(float);
        }
        
        return stats;
    }
    
    /**
     * @brief 设置配置
     */
    void SetConfig(const MemoryPoolConfig& config) {
        config_ = config;
    }
    
private:
    TensorMemoryPool() = default;
    
    ~TensorMemoryPool() {
        Clear();
    }
    
    // 禁止拷贝
    TensorMemoryPool(const TensorMemoryPool&) = delete;
    TensorMemoryPool& operator=(const TensorMemoryPool&) = delete;
    
    MemoryPoolConfig config_;
    mutable std::mutex mutex_;
    
    // used_: ptr -> size (elements)
    std::unordered_map<float*, size_t> used_;
    
    // free_blocks_: size -> ptr (ordered for best-fit)
    std::map<size_t, float*> free_blocks_;
};

/**
 * @brief 图像预处理缓冲区池
 */
class PreprocessBufferPool {
public:
    static PreprocessBufferPool& Instance() {
        static PreprocessBufferPool instance;
        return instance;
    }
    
    /**
     * @brief 获取标准640x640预处理缓冲区
     */
    float* GetResizeBuffer(int width = 640, int height = 640, int channels = 3) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        Key key{width, height, channels};
        auto& stack = pools_[key];
        
        if (!stack.empty()) {
            float* ptr = stack.back();
            stack.pop_back();
            return ptr;
        }
        
        // 创建新缓冲区
        size_t size = static_cast<size_t>(width) * height * channels;
        return static_cast<float*>(VE_ALIGNED_ALLOC(64, size * sizeof(float)));
    }
    
    /**
     * @brief 归还缓冲区
     */
    void Return(float* ptr, int width = 640, int height = 640, int channels = 3) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        Key key{width, height, channels};
        auto& stack = pools_[key];
        
        // 限制每个池的大小
        if (stack.size() < 10) {
            stack.push_back(ptr);
        } else {
            VE_ALIGNED_FREE(ptr);
        }
    }
    
private:
    PreprocessBufferPool() = default;
    ~PreprocessBufferPool() {
        for (auto& pool : pools_) {
            for (auto ptr : pool.second) {
                VE_ALIGNED_FREE(ptr);
            }
        }
    }
    
    struct Key {
        int width, height, channels;
        bool operator<(const Key& other) const {
            if (width != other.width) return width < other.width;
            if (height != other.height) return height < other.height;
            return channels < other.channels;
        }
    };
    
    std::mutex mutex_;
    std::map<Key, std::vector<float*>> pools_;
};

/**
 * @brief RAII包装器 - 自动内存管理
 */
template<typename T>
class PooledTensor {
public:
    PooledTensor() : ptr_(nullptr), size_(0) {}
    
    explicit PooledTensor(size_t size) {
        Allocate(size);
    }
    
    ~PooledTensor() {
        Deallocate();
    }
    
    // 移动构造
    PooledTensor(PooledTensor&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    PooledTensor& operator=(PooledTensor&& other) noexcept {
        if (this != &other) {
            Deallocate();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // 禁止拷贝
    PooledTensor(const PooledTensor&) = delete;
    PooledTensor& operator=(const PooledTensor&) = delete;
    
    T* Allocate(size_t size) {
        Deallocate();
        size_ = size;
        ptr_ = static_cast<T*>(VE_ALIGNED_ALLOC(64, size * sizeof(T)));
        return ptr_;
    }
    
    void Deallocate() {
        if (ptr_) {
            VE_ALIGNED_FREE(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }
    
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return size_; }
    
    T& operator[](size_t idx) { return ptr_[idx]; }
    const T& operator[](size_t idx) const { return ptr_[idx]; }
    
    T* release() {
        T* tmp = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return tmp;
    }
    
private:
    T* ptr_;
    size_t size_;
};

} // namespace vision_engine

#endif // VE_MEMORY_POOL_H
