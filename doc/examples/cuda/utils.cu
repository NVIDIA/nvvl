#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Not very optimized, but it's just for the test/example
__global__ void half2float_kernel(half* input, size_t input_pitch,
                                  uint16_t width, uint16_t height,
                                  float* output, size_t output_pitch) {
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    auto in = input + y*input_pitch + x;
    auto out = output + y*output_pitch + x;

    *reinterpret_cast<float2*>(out) =
            __half22float2(*reinterpret_cast<half2*>(in));
}

int divUp(int total, int grain) {
    return (total + grain - 1) / grain;
}

void half2float(half* input, size_t input_pitch,
                uint16_t width, uint16_t height,
                float* output, size_t output_pitch) {
    dim3 block(32, 8);
    dim3 grid(divUp(width/2, block.x), divUp(height, block.y));

    if (width % 2 == 1) {
        throw std::runtime_error("Width must be a multiple of 2.");
    }

    half2float_kernel<<<grid, block>>>
            (input, input_pitch, width, height, output, output_pitch);
    auto e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::cerr << "CUDA runtime error converting half to float: "
                  << cudaGetErrorString(e)
                  << std::endl;
    }
}
