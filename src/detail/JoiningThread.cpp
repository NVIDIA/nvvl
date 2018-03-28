#include <iostream>
#include <system_error>

#include "detail/JoiningThread.h"

namespace NVVL {
namespace detail {

JoiningThread::JoiningThread() : t_{} {
}

JoiningThread::JoiningThread(std::thread t) : t_{std::move(t)} {
}

JoiningThread::~JoiningThread() {
    if (t_.joinable()) {
        try {
            t_.join();
        } catch (const std::system_error& e) {
            std::cerr << "System error joining thread: "
                      << e.what() << std::endl;
        }
    }
}

}
}
