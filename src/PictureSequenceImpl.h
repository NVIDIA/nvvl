#pragma once

#include <boost/mp11.hpp>
#include <boost/variant.hpp>
#include <condition_variable>
#include <mutex>
#include <type_traits>
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
    CudaEvent(int device_id) : CudaEvent(device_id, cudaEventDisableTiming) {
    }

    CudaEvent(int device_id, unsigned int flags) : valid_{false}, flags_{flags} {
        int orig_device;
        cudaGetDevice(&orig_device);
        auto set_device = false;
        if (device_id >= 0 && orig_device != device_id) {
            set_device = true;
            cucall(cudaSetDevice(device_id));
        }
        valid_ = cucall(cudaEventCreateWithFlags(&event_, flags));
        if (set_device) {
            cucall(cudaSetDevice(orig_device));
        }
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

// map NVVL_Pic{Data,Meta}Types to real types

// boost::mp11 prepends all signals with "mp_" making this tolerable
using namespace boost::mp11;
template<NVVL_PicDataType I>
using mp_pdt = std::integral_constant<NVVL_PicDataType, I>;

using PDTMap = mp_list<
    mp_list<mp_pdt<PDT_BYTE>, uint8_t>,
    mp_list<mp_pdt<PDT_HALF>, half>,
    mp_list<mp_pdt<PDT_FLOAT>, float>
    >;
using PDTypes = mp_transform<mp_second, PDTMap>;

template<NVVL_PicMetaType I>
using mp_pmt = std::integral_constant<NVVL_PicMetaType, I>;

using PMTMap = mp_list<
    mp_list<mp_pmt<PMT_INT>, int>,
    mp_list<mp_pmt<PMT_STRING>, std::string>
    >;
using PMTypes = mp_transform<mp_second, PMTMap>;

class PictureSequence::impl {
  public:

    impl(uint16_t count, int device_id);

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
    template<class T> using add_layer = Layer<T>;
    using LayerVariant = mp_rename<mp_transform<add_layer, PDTypes>, boost::variant>;

    template<class T> using add_vector = std::vector<T>;
    using Meta = mp_rename<mp_transform<add_vector, PMTypes>, boost::variant>;

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
