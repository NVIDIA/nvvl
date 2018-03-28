#include <cuda_fp16.h>

void half2float(half* input, size_t input_pitch,
                uint16_t width, uint16_t height,
                float* output, size_t output_pitch);
