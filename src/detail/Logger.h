#pragma once

#include <unordered_map>
#include <fstream>
#include <iostream>
#include <sstream>

#include "VideoLoader.h"

namespace NVVL {
namespace detail {

class Logger {
  public:
    using ManipFn = std::ostream& (*)(std::ostream&);
    using FlagsFn = std::ios_base& (*)(std::ios_base&);

    Logger() {
    }

    Logger(LogLevel level) : level_{level}, null_stream_{}, out_{} {
        set_levels(level);
    }

    void set_output(LogLevel level, std::ostream& out) {
        out_.erase(level);
        out_.insert(std::make_pair(level, std::ref(out)));
    }

    void set_level(LogLevel level) {
        out_.clear();
        set_levels(level);
        level_ = level;
    }

    LogLevel get_level() {
        return level_;
    }

    std::ostream& debug() {
        return out_.at(LogLevel_Debug);
    }

    std::ostream& info() {
        return out_.at(LogLevel_Info);
    }

    std::ostream& warn() {
        return out_.at(LogLevel_Warn).get() << "\e[1mWARNING: \e[0m";
    }

    std::ostream& error() {
        return out_.at(LogLevel_Error).get() << "\e[1mERROR: \e[0m";
    }

  private:
    void set_levels(LogLevel level) {
        set_output(LogLevel_Debug, level, std::cout);
        set_output(LogLevel_Info, level, std::cout);
        set_output(LogLevel_Warn, level, std::cerr);
        set_output(LogLevel_Error, level, std::cerr);
    }

    void set_output(LogLevel level, LogLevel min, std::ostream& out) {
        if (level >= min) {
            out_.insert(std::make_pair(level, std::ref(out)));
        } else {
            out_.insert(std::make_pair(level, std::ref(null_stream_)));
        }
    }

    LogLevel level_;
    std::ofstream null_stream_;
    std::unordered_map<LogLevel,
                       std::reference_wrapper<std::ostream>,
                       std::hash<int>> out_;
};

extern Logger default_log;

}
}
