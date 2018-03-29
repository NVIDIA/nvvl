#pragma once

#include <string>

#include "PictureSequenceImpl.h"

class AVPacket;
#ifdef HAVE_AVSTREAM_CODECPAR
class AVCodecParameters;
using CodecParameters = AVCodecParameters;
#else
class AVCodecContext;
using CodecParameters = AVCodecContext;
#endif

namespace NVVL {

class FrameReq;
class PictureSequence;

namespace detail {

class Logger;

struct FrameReq {
    std::string filename;
    int frame;
    int count;
};

class CUStream {
  public:
    CUStream() {
        cucall(cudaStreamCreate(&stream_));
    }

    ~CUStream() {
        cucall(cudaStreamDestroy(stream_));
    }

    CUStream(const CUStream&) = delete;
    CUStream& operator=(const CUStream&) = delete;
    CUStream(CUStream&&) = delete;
    CUStream& operator=(CUStream&&) = delete;

    operator cudaStream_t() const {
        return stream_;
    }

  private:
    cudaStream_t stream_;
};

class Decoder {
  public:
    Decoder();
    Decoder(int device_id, Logger& logger,
            const CodecParameters* codecpar);
    Decoder(const Decoder&) = default;
    Decoder(Decoder&&) = default;
    Decoder& operator=(const Decoder&) = default;
    Decoder& operator=(Decoder&&) = default;
    virtual ~Decoder() = default;

    int decode_packet(AVPacket* pkt);

    virtual void push_req(FrameReq req);

    virtual void receive_frames(PictureSequence& sequence);

    virtual void finish();

  protected:
    virtual int decode_av_packet(AVPacket* pkt);

    void record_sequence_event_(PictureSequence& sequence);

    // We're buddies with PictureSequence so we can forward a visitor
    // on to it's private implementation.
    template<typename Visitor>
    void foreach_layer(PictureSequence& sequence, const Visitor& visitor) {
        sequence.pImpl->foreach_layer(visitor);
    }

    const int device_id_;
    CUStream stream_;
    const CodecParameters* codecpar_;

    detail::Logger& log_;
};


}
}
