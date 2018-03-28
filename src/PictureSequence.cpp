#include "PictureSequence.h"
#include "PictureSequenceImpl.h"

namespace NVVL {

PictureSequence::PictureSequence(uint16_t count)
    : pImpl{std::make_unique<impl>(count)}
{}

PictureSequence::impl::impl(uint16_t count)
    : layers_{}, meta_{}, started_{false}, event_{}, count_{count}
{}

PictureSequence::~PictureSequence() = default;
PictureSequence::PictureSequence(PictureSequence&&) = default;
PictureSequence& PictureSequence::operator=(PictureSequence&&) = default;

// these get instantiated for all the types we care about by the C interface
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

void nvvl_set_layer(PictureSequenceHandle sequence, const NVVL_PicLayer* layer, const char* name) {
    auto ps = reinterpret_cast<PictureSequence*>(sequence);

    switch(layer->type) {
        case PDT_NONE:
            std::cerr << "Layer type is not set" << std::endl;
            return;

        case PDT_BYTE:
            ps->set_layer<uint8_t>(name, layer);
            return;

        case PDT_HALF:
            ps->set_layer<half>(name, layer);
            return;

        case PDT_FLOAT:
            ps->set_layer<float>(name, layer);
            return;
    }
    std::cerr << "Unimplemented layer type" << std::endl;
}

void* nvvl_get_or_add_meta_array(PictureSequenceHandle sequence, NVVL_PicMetaType type, const char* name) {
    // TODO catch boost::get exception and return a valid error
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    switch(type) {
        case PMT_UINT8:
            return reinterpret_cast<void*>(ps->get_or_add_meta<uint8_t>(name).data());

        case PMT_UINT16:
            return reinterpret_cast<void*>(ps->get_or_add_meta<uint16_t>(name).data());

        case PMT_UINT32:
            return reinterpret_cast<void*>(ps->get_or_add_meta<uint32_t>(name).data());

        case PMT_INT:
            return reinterpret_cast<void*>(ps->get_or_add_meta<int>(name).data());

        case PMT_FLOAT:
            return reinterpret_cast<void*>(ps->get_or_add_meta<float>(name).data());

        case PMT_STRING:
            // Can't currently get an array of strings
            return nullptr;

        default:
            std::cerr << "Unimplemented meta array type" << std::endl;
            return nullptr;
    }
    return nullptr;
}

const void* nvvl_get_meta_array(PictureSequenceHandle sequence, NVVL_PicMetaType type, const char* name) {
    // couldn't figure out how to not duplicate switch above... can't
    // really have a pointer to a template member function...
    auto ps = reinterpret_cast<const PictureSequence*>(sequence);
    try {
        switch(type) {
            case PMT_UINT8:
                return reinterpret_cast<const void*>(ps->get_meta<uint8_t>(name).data());

            case PMT_UINT16:
                return reinterpret_cast<const void*>(ps->get_meta<uint16_t>(name).data());

            case PMT_UINT32:
                return reinterpret_cast<const void*>(ps->get_meta<uint32_t>(name).data());

            case PMT_INT:
                return reinterpret_cast<const void*>(ps->get_meta<int>(name).data());

            case PMT_FLOAT:
                return reinterpret_cast<const void*>(ps->get_meta<float>(name).data());

            case PMT_STRING:
                // Can't currently get an array of strings
                return nullptr;

            default:
                std::cerr << "Unimplemented meta array type" << std::endl;
                return nullptr;
        }
    } catch (const std::runtime_error&) {
        return nullptr;
    } catch (const boost::bad_get&) {
        std::cerr << "Tried to get wrong type from a sequence's meta array" << std::endl;
        return nullptr;
    }
    return nullptr;
}

const char* nvvl_get_meta_str(PictureSequenceHandle sequence, const char* name, int index) {
    // TODO catch boost::get exception and return a valid error
    auto ps = reinterpret_cast<const PictureSequence*>(sequence);
    return ps->get_meta<std::string>(name)[index].c_str();
}

int nvvl_get_sequence_count(PictureSequenceHandle sequence) {
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    return ps->count();
}

NVVL_PicLayer nvvl_get_layer(PictureSequenceHandle sequence, NVVL_PicDataType type, const char* name) {
    // TODO catch boost::get exception and return a valid error
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    NVVL_PicLayer ret;
    ret.type = type;

    // here we convert back from statically typed C++ to dynamically typed C
    auto convert = [&](const auto& l) {
        ret.desc = l.desc;
        ret.index_map = l.index_map.data();
        ret.data = reinterpret_cast<void*>(l.data);
    };

    switch(type) {
        case PDT_BYTE:
            convert(ps->get_layer<uint8_t>(name));
            break;

        case PDT_HALF:
            convert(ps->get_layer<half>(name));
            break;

        case PDT_FLOAT:
            convert(ps->get_layer<float>(name));
            break;

        default:
            std::cerr << "Unimplemented layer type" << std::endl;
            ret.data = nullptr;
            break;
    }
    return ret;
}

NVVL_PicLayer nvvl_get_layer_indexed(PictureSequenceHandle sequence, NVVL_PicDataType type,
                                     const char* name, int index) {
    // TODO catch boost::get exception and return a valid error
    auto ps = reinterpret_cast<PictureSequence*>(sequence);
    NVVL_PicLayer ret;
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

    switch(type) {
        case PDT_BYTE:
            convert(ps->get_layer<uint8_t>(name, index));
            break;

        case PDT_HALF:
            convert(ps->get_layer<half>(name, index));
            break;

        case PDT_FLOAT:
            convert(ps->get_layer<float>(name, index));
            break;

        default:
            std::cerr << "Unimplemented layer type" << std::endl;
            ret.data = nullptr;
            break;
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
