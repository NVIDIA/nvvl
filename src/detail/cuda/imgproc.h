#pragma once

#include "PictureSequence.h"

namespace NVVL {
namespace detail {

template<typename T>
void process_frame(
    cudaTextureObject_t chroma, cudaTextureObject_t luma,
    const PictureSequence::Layer<T>& output, int index, cudaStream_t stream,
    uint16_t input_width, uint16_t input_height);

}
}
