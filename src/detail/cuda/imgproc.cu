#include "detail/cuda/imgproc.h"

#include "PictureSequence.h"
#include "detail/utils.h"

#include <cuda_fp16.h>

namespace NVVL {
namespace detail {

namespace {

// using math from https://msdn.microsoft.com/en-us/library/windows/desktop/dd206750(v=vs.85).aspx

template<typename T>
struct yuv {
    T y, u, v;
};

__constant__ float yuv2rgb_mat[9] = {
    1.164383f,  0.0f,       1.596027f,
    1.164383f, -0.391762f, -0.812968f,
    1.164383f,  2.017232f,  0.0f
};

__device__ float clip(float x, float max) {
    return fmin(fmax(x, 0.0f), max);
}

template<typename T>
__device__ T convert(const float x) {
    return static_cast<T>(x);
}

template<>
__device__ half convert<half>(const float x) {
    return __float2half(x);
}

template<>
__device__ uint8_t convert<uint8_t>(const float x) {
    return static_cast<uint8_t>(roundf(x));
}

template<typename YUV_T, typename RGB_T>
__device__ void yuv2rgb(const yuv<YUV_T>& yuv, RGB_T* rgb,
                        size_t stride, bool normalized) {
    auto mult = normalized ? 1.0f : 255.0f;
    auto y = (static_cast<float>(yuv.y) - 16.0f/255) * mult;
    auto u = (static_cast<float>(yuv.u) - 128.0f/255) * mult;
    auto v = (static_cast<float>(yuv.v) - 128.0f/255) * mult;

    auto& m = yuv2rgb_mat;

    // could get tricky with a lambda, but this branch seems faster
    float r, g, b;
    if (normalized) {
        r = clip(y*m[0] + u*m[1] + v*m[2], 1.0);
        g = clip(y*m[3] + u*m[4] + v*m[5], 1.0);
        b = clip(y*m[6] + u*m[7] + v*m[8], 1.0);
    } else {
        r = clip(y*m[0] + u*m[1] + v*m[2], 255.0);
        g = clip(y*m[3] + u*m[4] + v*m[5], 255.0);
        b = clip(y*m[6] + u*m[7] + v*m[8], 255.0);
    }

    rgb[0] = convert<RGB_T>(r);
    rgb[stride] = convert<RGB_T>(g);
    rgb[stride*2] = convert<RGB_T>(b);
}

template<typename T>
__global__ void process_frame_kernel(
    cudaTextureObject_t luma, cudaTextureObject_t chroma,
    PictureSequence::Layer<T> dst, int index,
    uint16_t scale_width, float fx, float fy) {

    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst.desc.width || dst_y >= dst.desc.height)
        return;

    auto src_x = 0.0f;
    if (dst.desc.horiz_flip) {
        src_x = (scale_width - dst.desc.crop_x - dst_x) * fx;
    } else {
        src_x = (dst.desc.crop_x + dst_x) * fx;
    }

    auto src_y = static_cast<float>(dst_y + dst.desc.crop_y) * fy;

    yuv<float> yuv;
    yuv.y = tex2D<float>(luma, src_x + 0.5, src_y + 0.5);
    auto uv = tex2D<float2>(chroma, (src_x / 2) + 0.5, (src_y / 2) + 0.5);
    yuv.u = uv.x;
    yuv.v = uv.y;

    auto out = &dst.data[dst_x * dst.desc.stride.x +
                         dst_y * dst.desc.stride.y +
                         index * dst.desc.stride.n];

    switch(dst.desc.color_space) {
        case ColorSpace_RGB:
            yuv2rgb(yuv, out, dst.desc.stride.c, dst.desc.normalized);
            break;

        case ColorSpace_YCbCr:
            auto mult = dst.desc.normalized ? 1.0f : 255.0f;
            out[0] = convert<T>(yuv.y * mult);
            out[dst.desc.stride.c] = convert<T>(yuv.u * mult);
            out[dst.desc.stride.c*2] = convert<T>(yuv.v * mult);
            break;
    };
}

int divUp(int total, int grain) {
    return (total + grain - 1) / grain;
}

} // anon namespace

template<typename T>
void process_frame(
    cudaTextureObject_t chroma, cudaTextureObject_t luma,
    const PictureSequence::Layer<T>& output, int index, cudaStream_t stream,
    uint16_t input_width, uint16_t input_height) {

    if (!(std::is_same<T, half>::value || std::is_floating_point<T>::value)
        && output.desc.normalized) {
        throw std::runtime_error("Output must be floating point to be normalized.");
    }

    auto scale_width = output.desc.scale_width > 0 ? output.desc.scale_width : input_width;
    auto scale_height = output.desc.scale_height > 0 ? output.desc.scale_height : input_height;

    auto fx = static_cast<float>(input_width) / scale_width;
    auto fy = static_cast<float>(input_height) / scale_height;

    dim3 block(32, 8);
    dim3 grid(divUp(output.desc.width, block.x), divUp(output.desc.height, block.y));

    process_frame_kernel<<<grid, block, 0, stream>>>
            (luma, chroma, output, index, scale_width, fx, fy);
}

template void process_frame<uint8_t>(
    cudaTextureObject_t chroma, cudaTextureObject_t luma,
    const PictureSequence::Layer<uint8_t>& output, int index, cudaStream_t stream,
    uint16_t input_width, uint16_t input_height);

template void process_frame<half>(
    cudaTextureObject_t chroma, cudaTextureObject_t luma,
    const PictureSequence::Layer<half>& output, int index, cudaStream_t stream,
    uint16_t input_width, uint16_t input_height);

template void process_frame<float>(
    cudaTextureObject_t chroma, cudaTextureObject_t luma,
    const PictureSequence::Layer<float>& output, int index, cudaStream_t stream,
    uint16_t input_width, uint16_t input_height);

} // namespace detail
} // namespace NVVL
