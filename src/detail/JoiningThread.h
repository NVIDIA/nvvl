#pragma once

#include <thread>

namespace NVVL {
namespace detail {

class JoiningThread {
  public:
    JoiningThread();
    JoiningThread(std::thread t);
    ~JoiningThread();
    JoiningThread(const JoiningThread& other) = delete;
    JoiningThread& operator=(const JoiningThread& other) = delete;
    JoiningThread(JoiningThread&& other) = default;
    JoiningThread& operator=(JoiningThread&& other) = default;
  private:
    std::thread t_;
};

}
}
