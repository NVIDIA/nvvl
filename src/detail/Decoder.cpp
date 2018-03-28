#include <libavcodec/avcodec.h>

#include "VideoLoader.h"
#include "detail/Logger.h"
#include "detail/Decoder.h"

namespace NVVL {
namespace detail {

Logger default_log;

Decoder::Decoder() : device_id_{0}, stream_{}, codecpar_{}, log_{default_log}
{
}

Decoder::Decoder(int device_id, Logger& logger,
                 const AVCodecParameters* codecpar)
    : device_id_{device_id}, stream_{}, codecpar_{codecpar}, log_{logger}
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

// This has to be here since Decoder is the only friend of PictureSequence
void Decoder::record_sequence_event_(PictureSequence& sequence) {
    sequence.pImpl->event_.record(stream_);
    sequence.pImpl->set_started_(true);
}

}
}
