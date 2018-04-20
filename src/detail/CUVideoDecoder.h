#pragma once

#include "nvcuvid/nvcuvid.h"

namespace NVVL {

class Logger;

namespace detail {

class CUVideoDecoder {
  public:
    CUVideoDecoder();
    CUVideoDecoder(Logger& log);
    CUVideoDecoder(Logger& log, CUvideodecoder);
    ~CUVideoDecoder();

    // no copying
    CUVideoDecoder(const CUVideoDecoder&) = delete;
    CUVideoDecoder& operator=(const CUVideoDecoder&) = delete;

    CUVideoDecoder(CUVideoDecoder&& other);
    CUVideoDecoder& operator=(CUVideoDecoder&& other);

    operator CUvideodecoder() const;

    int initialize(CUVIDEOFORMAT* format);
    bool initialized() const;

    uint16_t width() const;
    uint16_t height() const;

  private:
    Logger* log_;
    CUvideodecoder decoder_;
    CUVIDDECODECREATEINFO decoder_info_;

    bool initialized_;
};


}
}
