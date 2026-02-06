#ifndef VE_THREAD_POOL_H
#define VE_THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace vision_engine {

/**
 * @brief 线程池
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();
    
    // 禁止拷贝
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    
    // 添加任务
    void Enqueue(std::function<void()> task);
    
    // 获取待执行任务数量
    size_t GetTaskCount() const;
    
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

} // namespace vision_engine

#endif // VE_THREAD_POOL_H
