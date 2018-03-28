#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace NVVL {
namespace detail {

template<typename T>
class Queue {
  public:
    Queue() : interrupt_{false} {}

    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(lock_);
            queue_.push(std::move(item));
        }
        cond_.notify_one();
    }

    T pop() {
        static auto int_return = T{};
        std::unique_lock<std::mutex> lock{lock_};
        cond_.wait(lock, [&](){return !queue_.empty() || interrupt_;});
        if (interrupt_) {
            return std::move(int_return);
        }
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    const T& peek() {
        static auto int_return = T{};
        std::unique_lock<std::mutex> lock{lock_};
        cond_.wait(lock, [&](){return !queue_.empty() || interrupt_;});
        if (interrupt_) {
            return std::move(int_return);
        }
        return queue_.front();
    }

    bool empty() const {
        return queue_.empty();
    }

    typename std::queue<T>::size_type size() const {
        return queue_.size();
    }

    void cancel_pops() {
        interrupt_ = true;
        cond_.notify_all();
    }

  private:
    std::queue<T> queue_;
    std::mutex lock_;
    std::condition_variable cond_;
    std::atomic<bool> interrupt_;
};

}
}
