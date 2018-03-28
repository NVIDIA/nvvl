#pragma once

#include <boost/variant.hpp>
#include <condition_variable>
#include <mutex>
#include <unordered_map>

#include <cuda_fp16.h>

#include "PictureSequence.h"
#include "detail/utils.h"

namespace NVVL {

namespace detail {
class Decoder;
}

class CudaEvent {
  public:
    CudaEvent() : CudaEvent(cudaEventDisableTiming) {
    }

    CudaEvent(unsigned int flags) : valid_{false}, flags_{flags} {
        valid_ = cucall(cudaEventCreateWithFlags(&event_, flags));
    }

    ~CudaEvent() {
        if (valid_) {
            cucall(cudaEventDestroy(event_));
        }
    }

    CudaEvent(CudaEvent&& other)
        : valid_{other.valid_}, flags_{other.flags_},
          event_{other.event_} {
        other.event_ = 0;
        other.valid_ = false;
    }

    CudaEvent& operator=(CudaEvent&& other) {
        if (valid_) {
            cucall(cudaEventDestroy(event_));
        }
        valid_ = other.valid_;
        flags_ = other.flags_;
        event_ = other.event_;
        other.event_ = 0;
        other.valid_ = false;
        return *this;
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(CudaEvent&) = delete;

    operator cudaEvent_t() const {
        return event_;
    }

    void record(cudaStream_t stream) {
        cucall(cudaEventRecord(event_, stream));
    }

  private:
    bool valid_;
    unsigned int flags_;
    cudaEvent_t event_;
};

class PictureSequence::impl {
  public:

    impl(uint16_t count);

    template<typename T>
    void set_layer(std::string name, const Layer<T>& layer) {
        auto l = LayerVariant{layer};
        auto r = layers_.emplace(std::move(name), std::move(l));
        if (!r.second) {
            throw std::runtime_error("Setting a layer that already exists");
        }
    }

    template<typename T>
    Layer<T> get_layer(std::string name, int index) const {
        auto l = layers_.find(name);
        if (l == layers_.end()) {
            throw std::runtime_error("Requested a layer that doesn't exist");
        }
        auto ret = boost::get<Layer<T>>(l->second);

        ret.data = ret.data + (ret.desc.stride.n * index);
        return ret;
    }

    template<typename T>
    const Layer<T>& get_layer(std::string name) const {
        auto l = layers_.find(name);
        if (l == layers_.end()) {
            throw std::runtime_error("Requested a layer that doesn't exist");
        }
        return boost::get<Layer<T>>(l->second);
    }

    template<typename Visitor>
    void foreach_layer(const Visitor& visitor) {
        for(auto&& l : layers_) {
            boost::apply_visitor(visitor, l.second);
        }
    }

    bool has_layer(std::string name) const {
        return layers_.count(name) > 0;
    }

    template<typename T>
    std::vector<T>& get_or_add_meta(std::string name) {
        auto m = meta_.find(name);
        if (m == meta_.end()) {
            auto v = std::vector<T>(count_);
            m = meta_.emplace(std::move(name), std::move(v)).first;
        }
        return boost::get<std::vector<T>>(m->second);
    }

    template<typename T>
    const std::vector<T>& get_meta(std::string name) const {
        auto m = meta_.find(name);
        if (m == meta_.end()) {
            throw std::runtime_error(std::string("Unable to find metadata ") + name);
        }
        return boost::get<std::vector<T>>(m->second);
    }

    template<typename Visitor>
    typename Visitor::result_type
    visit_meta(const Visitor& visitor, std::string name) const {
        auto m = meta_.find(name);
        if (m == meta_.end()) {
            return;
        }
        return boost::apply_visitor(visitor, m->second);
    }

    bool has_meta(std::string name) const;

    int count() const;

    void set_count(int count);

    void wait() const;
    void wait(cudaStream_t stream) const;

  private:
    using LayerVariant = boost::variant<Layer<uint8_t>,
                                        Layer<half>,
                                        Layer<float>>;

    using Meta = boost::variant<std::vector<int>,
                                std::vector<uint8_t>,
                                std::vector<uint16_t>,
                                std::vector<uint32_t>,
                                std::vector<float>,
                                std::vector<const void*>,
                                std::vector<std::string>>;

    std::unordered_map<std::string, LayerVariant> layers_;
    std::unordered_map<std::string, Meta> meta_;

    void set_started_(bool started);

    mutable std::mutex started_lock_;
    mutable std::condition_variable started_cv_;
    bool started_;
    CudaEvent event_;
    uint16_t count_;

    void wait_until_started_() const;

    // Decoder's needs to record the event and indicate transfer has started
    friend class detail::Decoder;
};

}
