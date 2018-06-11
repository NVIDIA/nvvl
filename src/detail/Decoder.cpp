#include <libavcodec/avcodec.h>

#include "VideoLoader.h"
#include "detail/Logger.h"
#include "detail/Decoder.h"

namespace NVVL {
namespace detail {

Logger default_log;

CUStream::CUStream(int device_id, bool default_stream) : created_{false}, stream_{0} {
    if (!default_stream) {
        int orig_device;
        cudaGetDevice(&orig_device);
        auto set_device = false;
        if (device_id >= 0 && orig_device != device_id) {
            set_device = true;
            cudaSetDevice(device_id);
        }
        cucall(cudaStreamCreate(&stream_));
        created_ = true;
        if (set_device) {
            cucall(cudaSetDevice(orig_device));
        }
    }
}

CUStream::~CUStream() {
    if (created_) {
        cucall(cudaStreamDestroy(stream_));
    }
}

CUStream::CUStream(CUStream&& other)
    : created_{other.created_}, stream_{other.stream_}
{
    other.stream_ = 0;
    other.created_ = false;
}

CUStream& CUStream::operator=(CUStream&& other) {
    stream_ = other.stream_;
    created_ = other.created_;
    other.stream_ = 0;
    other.created_ = false;
    return *this;
}


CUStream::operator cudaStream_t() {
    return stream_;
}

Decoder::Decoder() : device_id_{-1}, stream_{-1, true}, codecpar_{}, log_{default_log}
{
}

Decoder::Decoder(int device_id, Logger& logger,
                 const CodecParameters* codecpar)
    : device_id_{device_id}, stream_{device_id, false}, codecpar_{codecpar}, log_{logger}
{
}

int Decoder::decode_packet(AVPacket* pkt) {
    switch(codecpar_->codec_type) {
        case AVMEDIA_TYPE_AUDIO:
        case AVMEDIA_TYPE_VIDEO:
            return decode_av_packet(pkt);

        default:
            throw std::runtime_error("Got to decode_packet in a decoder that is not "
                                     "for an audio, video, or subtitle stream.");
    }
    return -1;
}

void Decoder::push_req(FrameReq req) {
    throw std::runtime_error("Decoding audio/video data is not implemented for this decoder.");
}

void Decoder::receive_frames(PictureSequence& sequence) {
    throw std::runtime_error("Decoding audio/video data is not implemented for this decoder.");
}

int Decoder::decode_av_packet(AVPacket* pkt) {
    throw std::runtime_error("Decoding audio/video data is not implemented for this decoder.");
    return -1;
}

void Decoder::finish() {
    // Children will have to override if they want to do something
}

void Decoder::use_default_stream() {
    stream_ = CUStream{device_id_, true};
}

// This has to be here since Decoder is the only friend of PictureSequence
void Decoder::record_sequence_event_(PictureSequence& sequence) {
    sequence.pImpl->event_.record(stream_);
    sequence.pImpl->set_started_(true);
}

}
}
