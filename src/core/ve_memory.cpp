#include "ve_memory.h"
#include <cstdlib>
#include <cstring>
#include <new>

namespace vision_engine {

// 内存池实现
MemoryPool::MemoryPool(size_t block_size, size_t num_blocks)
    : block_size_(block_size), num_blocks_(num_blocks) {
    pool_size_ = block_size_ * num_blocks_;
    
    // 对齐分配
    pool_ = AlignedAlloc(pool_size_, alignment_);
    if (!pool_) {
        throw std::bad_alloc();
    }
    
    // 初始化空闲链表
    free_list_ = static_cast<void**>(pool_);
    for (size_t i = 0; i < num_blocks_ - 1; ++i) {
        free_list_[i] = static_cast<char*>(pool_) + (i + 1) * block_size_;
        free_list_[i] = nullptr;  // 标记为可用
    }
    free_list_[num_blocks_ - 1] = nullptr;
    free_count_ = num_blocks_;
}

MemoryPool::~MemoryPool() {
    if (pool_) {
        AlignedFree(pool_);
        pool_ = nullptr;
    }
}

void* MemoryPool::Allocate() {
    if (free_count_ == 0) {
        return nullptr;
    }
    void* block = free_list_[--free_count_];
    return block;
}

void MemoryPool::Deallocate(void* ptr) {
    if (!ptr || free_count_ >= num_blocks_) {
        return;
    }
    free_list_[free_count_++] = ptr;
}

void MemoryPool::Reset() {
    free_count_ = num_blocks_;
    for (size_t i = 0; i < num_blocks_ - 1; ++i) {
        free_list_[i] = static_cast<char*>(pool_) + (i + 1) * block_size_;
    }
    free_list_[num_blocks_ - 1] = nullptr;
}

// 内存管理器实现
MemoryManager& MemoryManager::Instance() {
    static MemoryManager instance;
    return instance;
}

void* MemoryManager::AllocAligned(size_t size, size_t alignment) {
    return AlignedAlloc(size, alignment);
}

void MemoryManager::FreeAligned(void* ptr) {
    AlignedFree(ptr);
}

MemoryPool* MemoryManager::CreatePool(size_t block_size, size_t num_blocks) {
    auto pool = std::make_unique<MemoryPool>(block_size, num_blocks);
    MemoryPool* raw_ptr = pool.get();
    pools_.push_back(std::move(pool));
    return raw_ptr;
}

void MemoryManager::DestroyPool(MemoryPool* pool) {
    for (auto it = pools_.begin(); it != pools_.end(); ++it) {
        if (it->get() == pool) {
            pools_.erase(it);
            return;
        }
    }
}

size_t MemoryManager::GetUsedMemory() const {
    return used_memory_;
}

size_t MemoryManager::GetAllocatedMemory() const {
    return allocated_memory_;
}

size_t MemoryManager::GetPeakMemory() const {
    return peak_memory_;
}

} // namespace vision_engine
