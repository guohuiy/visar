#ifndef VE_MEMORY_H
#define VE_MEMORY_H

#include <cstddef>
#include <memory>
#include <vector>

#ifdef __cplusplus
namespace vision_engine {

/**
 * @brief 内存对齐工具
 */
inline void* AlignedAlloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(size, alignment);
#elif defined(__GNUC__) || defined(__clang__)
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
#endif
    return ptr;
}

inline void AlignedFree(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#elif defined(__GNUC__) || defined(__clang__)
    free(ptr);
#endif
}

/**
 * @brief 简单内存池
 */
class MemoryPool {
public:
    MemoryPool(size_t block_size, size_t num_blocks);
    ~MemoryPool();
    
    void* Allocate();
    void Deallocate(void* ptr);
    void Reset();
    
    size_t GetBlockSize() const { return block_size_; }
    size_t GetNumBlocks() const { return num_blocks_; }
    size_t GetFreeCount() const { return free_count_; }
    size_t GetTotalSize() const { return pool_size_; }
    
private:
    size_t block_size_;
    size_t num_blocks_;
    size_t pool_size_;
    size_t alignment_ = 16;
    void* pool_ = nullptr;
    void** free_list_ = nullptr;
    size_t free_count_ = 0;
};

/**
 * @brief 全局内存管理器
 */
class MemoryManager {
public:
    static MemoryManager& Instance();
    
    void* AllocAligned(size_t size, size_t alignment = 16);
    void FreeAligned(void* ptr);
    
    MemoryPool* CreatePool(size_t block_size, size_t num_blocks);
    void DestroyPool(MemoryPool* pool);
    
    size_t GetUsedMemory() const;
    size_t GetAllocatedMemory() const;
    size_t GetPeakMemory() const;
    
private:
    MemoryManager() = default;
    ~MemoryManager() = default;
    
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    std::vector<std::unique_ptr<MemoryPool>> pools_;
    size_t used_memory_ = 0;
    size_t allocated_memory_ = 0;
    size_t peak_memory_ = 0;
};

} // namespace vision_engine
#endif // __cplusplus

#endif // VE_MEMORY_H
