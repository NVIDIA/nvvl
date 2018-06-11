#include "PictureSequence.h"
#include "PictureSequenceImpl.h"

namespace NVVL {

PictureSequence::PictureSequence(uint16_t count)
    : PictureSequence(count, -1)
{}

PictureSequence::PictureSequence(uint16_t count, int device_id)
    : pImpl{std::make_unique<impl>(count, device_id)}
{}

PictureSequence::impl::impl(uint16_t count, int device_id)
    : layers_{}, meta_{}, started_{false}, event_{device_id}, count_{count}
{}

PictureSequence::~PictureSequence() = default;
PictureSequence::PictureSequence(PictureSequence&&) = default;
PictureSequence& PictureSequence::operator=(PictureSequence&&) = default;

template<typename T>
void PictureSequence::set_layer(std::string name, const PictureSequence::Layer<T>& layer) {
    pImpl->set_layer(name, layer);
}

template<typename T>
PictureSequence::Layer<T> PictureSequence::get_layer(std::string name, int index) const {
    return pImpl->get_layer<T>(name, index);
}

template<typename T>
const PictureSequence::Layer<T>& PictureSequence::get_layer(std::string name) const {
    return pImpl->get_layer<T>(name);
}

// need to explicitely instantiate these even though they are used by
// the C interface since g++ with -O2 will inline them in the C
// functions and not export the symbols

// instantiation code based on:
// https://stackoverflow.com/questions/5715586/how-to-explicitly-instantiate-a-template-for-all-members-of-mpl-vector-in-c/35895553
// but tweaked a bit rewritten with mp11. Basically we make a pointer
// to each function we want and that forces an instantiation. Not very
// pretty or readable, but makes it so we only have to write the types
// out once in PictureSequenceImpl.h
namespace {
template<class L> struct mp_instantiate_layer_funcs;
template<template<class...> class L> struct mp_instantiate_layer_funcs<L<>> {};
template<template<class...> class L, class T1, class... T>
struct mp_instantiate_layer_funcs<L<T1, T...>> {
    using PS = PictureSequence;
    mp_instantiate_layer_funcs() :
        sl{&PS::set_layer<T1>}, gl{&PS::get_layer<T1>}, gl2{&PS::get_layer<T1>} {}
    void (PS::*sl)(std::string, const PS::Layer<T1>&);
    PS::Layer<T1> (PS::*gl)(std::string, int) const;
    const PS::Layer<T1>& (PS::*gl2)(std::string) const;
    mp_instantiate_layer_funcs<L<T...>> next;
};
static mp_instantiate_layer_funcs<PDTypes> layer_funcs;
} // end anon namespace

bool PictureSequence::has_layer(std::string name) const {
    return pImpl->has_layer(name);
}

template<typename T>
std::vector<T>& PictureSequence::get_or_add_meta(std::string name) {
    return pImpl->get_or_add_meta<T>(name);
}

template<typename T>
const std::vector<T>& PictureSequence::get_meta(std::string name) const {
    return pImpl->get_meta<T>(name);
}

// do the same as above for meta funcs
namespace {
template<class L> struct mp_instantiate_meta_funcs;
template<template<class...> class L> struct mp_instantiate_meta_funcs<L<>> {};
template<template<class...> class L, class T1, class... T>
struct mp_instantiate_meta_funcs<L<T1, T...>> {
    using PS = PictureSequence;
    mp_instantiate_meta_funcs() :
        goam{&PS::get_or_add_meta<T1>}, gm{&PS::get_meta<T1>} {}
    std::vector<T1>& (PS::*goam)(std::string);
    const std::vector<T1>& (PS::*gm)(std::string) const;
    mp_instantiate_meta_funcs<L<T...>> next;
};
static mp_instantiate_meta_funcs<PMTypes> meta_funcs;
} // end anon namespace

bool PictureSequence::has_meta(std::string name) const {
    return pImpl->has_meta(name);
}

bool PictureSequence::impl::has_meta(std::string name) const {
    return meta_.count(name) > 0;
}

int PictureSequence::count() const {
    return pImpl->count();
}

int PictureSequence::impl::count() const {
    return count_;
}

void PictureSequence::set_count(int count) {
    pImpl->set_count(count);
}

void PictureSequence::impl::set_count(int count) {
    for (auto& m : meta_) {
        boost::apply_visitor([=](auto& v) {v.resize(count);}, m.second);
    }
    count_ = count;
}

void PictureSequence::impl::set_started_(bool started) {
    std::unique_lock<std::mutex> lock{started_lock_};
    started_ = started;
    lock.unlock();
    started_cv_.notify_one();
}

void PictureSequence::impl::wait_until_started_() const {
    std::unique_lock<std::mutex> lock{started_lock_};
    started_cv_.wait(lock, [&](){return started_;});
}

void PictureSequence::wait() const {
    pImpl->wait();
}

void PictureSequence::impl::wait() const {
    wait_until_started_();
    cucall(cudaEventSynchronize(event_));
}

void PictureSequence::wait(cudaStream_t stream) const {
    pImpl->wait(stream);
}

void PictureSequence::impl::wait(cudaStream_t stream) const {
    wait_until_started_();
    cucall(cudaStreamWaitEvent(stream, event_, 0));
}

} // namespace NVVL

extern "C" {

using PictureSequence = NVVL::PictureSequence;

PictureSequenceHandle nvvl_create_sequence(uint16_t count) {
    auto ps = new PictureSequence{count};
    return reinterpret_cast<PictureSequenceHandle>(ps);
}

PictureSequenceHandle nvvl_create_sequence_device(uint16_t count, int device_id) {
    auto ps = new PictureSequence{count, device_id};
    return reinterpret_cast<PictureSequenceHandle>(ps);
}

void* nvvl_get_or_add_meta_array(PictureSequenceHandle sequence,
                                 NVVL_PicMetaType type, const char* name) {
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    void* res = nullptr;
    using namespace boost::mp11;
    try {
        mp_for_each<NVVL::PMTMap>([&](auto P) -> void {
                if(type == mp_first<decltype(P)>::value) {
                    using T = mp_second<decltype(P)>;
                    res = reinterpret_cast<void*>(ps->get_or_add_meta<T>(name).data());
                }
            });
    } catch (const boost::bad_get&) {
        std::cerr << "Tried to get wrong type from a sequence's meta array" << std::endl;
        return nullptr;
    }
    if (!res) {
        std::cerr << "Unimplemented meta array type" << std::endl;
    }
    return res;
}

class not_implemented : public std::exception {
  public:
    virtual const char* what() const throw () {
        return "Functionality not yet implemented.";
    }
};

const void* nvvl_get_meta_array(PictureSequenceHandle sequence,
                                NVVL_PicMetaType type, const char* name) {
    auto ps = reinterpret_cast<const PictureSequence*>(sequence);
    const void* res = nullptr;
    using namespace boost::mp11;
    if (type == PMT_STRING) {
        // Can't currently get an array of strings, shouldn't be too
        // bad to implement if necessary, but I believe would require
        // allocating some memory, which would be unfortunate
        throw not_implemented();
    }
    try {
        mp_for_each<NVVL::PMTMap>([&](auto P) -> void {
                if(type == mp_first<decltype(P)>::value) {
                    using T = mp_second<decltype(P)>;
                    res = reinterpret_cast<const void*>(ps->get_meta<T>(name).data());
                }
            });
    } catch (const std::runtime_error&) {
        return nullptr;
    } catch (const boost::bad_get&) {
        std::cerr << "Tried to get wrong type from a sequence's meta array" << std::endl;
        return nullptr;
    }
    if (!res) {
        std::cerr << "Unimplemented meta array type" << std::endl;
    }
    return res;
}

const char* nvvl_get_meta_str(PictureSequenceHandle sequence,
                              const char* name, int index) {
    // TODO catch boost::get exception and return a valid error
    auto ps = reinterpret_cast<const PictureSequence*>(sequence);
    return ps->get_meta<std::string>(name)[index].c_str();
}

int nvvl_get_sequence_count(PictureSequenceHandle sequence) {
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    return ps->count();
}

void nvvl_set_layer(PictureSequenceHandle sequence,
                    const NVVL_PicLayer* layer, const char* name) {
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    if (layer->type == PDT_NONE) {
        std::cerr << "Layer type is not set" << std::endl;
        return;
    }
    auto set = false;
    using namespace boost::mp11;
    mp_for_each<NVVL::PDTMap>([&](auto P) -> void {
            if (!set && layer->type == mp_first<decltype(P)>::value) {
                using T = mp_second<decltype(P)>;
                ps->set_layer<T>(name, layer);
                set = true;
            }
        });
    if (!set) {
        std::cerr << "Unimplemented layer type" << std::endl;
    }
}

NVVL_PicLayer nvvl_get_layer(PictureSequenceHandle sequence,
                             NVVL_PicDataType type, const char* name) {
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    auto ret = NVVL_PicLayer{};
    ret.type = type;

    // here we convert back from statically typed C++ to dynamically typed C
    auto convert = [&](const auto& l) {
        ret.desc = l.desc;
        ret.index_map = l.index_map.data();
        ret.data = reinterpret_cast<void*>(l.data);
    };
    using namespace boost::mp11;
    try {
        mp_for_each<NVVL::PDTMap>([&](auto P) -> void {
                if (type == mp_first<decltype(P)>::value) {
                    using T = mp_second<decltype(P)>;
                    convert(ps->get_layer<T>(name));
            }
        });
    } catch (const boost::bad_get&) {
        std::cerr << "Tried to get wrong type from a sequence's layer" << std::endl;
    }
    if (!ret.data) {
        std::cerr << "Unimplemented layer type" << std::endl;
    }
    return ret;
}

NVVL_PicLayer nvvl_get_layer_indexed(PictureSequenceHandle sequence, NVVL_PicDataType type,
                                     const char* name, int index) {
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    auto ret = NVVL_PicLayer{};
    ret.type = type;

    // here we convert back from statically typed C++ to dynamically typed C
    auto convert = [&](const auto& l) {
        ret.desc = l.desc;

        // can't return a pointer to the temporary Layer<T>'s vector
        // data and we don't want to malloc something here and have
        // vague ownership, so we just don't return the index_map for
        // this case.
        ret.index_map = nullptr;
        ret.data = reinterpret_cast<void*>(l.data);
    };
    using namespace boost::mp11;
    try {
        mp_for_each<NVVL::PDTMap>([&](auto P) -> void {
                if (type == mp_first<decltype(P)>::value) {
                    using T = mp_second<decltype(P)>;
                    convert(ps->get_layer<T>(name, index));
            }
        });
    } catch (const boost::bad_get&) {
        std::cerr << "Tried to get wrong type from a sequence's layer" << std::endl;
    }
    if (!ret.data) {
        std::cerr << "Unimplemented layer type" << std::endl;
    }
    return ret;
}

void nvvl_sequence_wait(PictureSequenceHandle sequence) {
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    ps->wait();
}

void nvvl_sequence_stream_wait(PictureSequenceHandle sequence, cudaStream_t stream) {
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    ps->wait(stream);
}

void nvvl_free_sequence(PictureSequenceHandle sequence) {
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    delete ps;
}

}
