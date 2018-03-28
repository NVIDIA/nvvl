#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "VideoLoader.h"

void write_frames(PictureSequenceHandle sequence) {
    struct NVVL_PicLayer pixels = nvvl_get_layer(sequence, PDT_BYTE, "pixels");
    printf("Got a sequence of size: %d\n", pixels.desc.count);
    for (int i = 0; i < pixels.desc.count; i++) {
        uint8_t* p = pixels.data + i*pixels.desc.stride.n;
        uint8_t tmp[100];
        if (cudaMemcpy(tmp, p, 100, cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Couldn't copy frame data to cpu\n");
            exit(1);
        }

        uint32_t sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += tmp[i];
        }

        printf(" Frame %d sum (first 100): %d\n", i, sum);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <video file>\n", argv[0]);
        return -1;
    }

    const uint16_t sequence_width = 400;
    const uint16_t sequence_height = 400;
    const uint16_t sequence_count = 10;

    VideoLoaderHandle loader = nvvl_create_video_loader(0);

    PictureSequenceHandle sequence = nvvl_create_sequence(sequence_count);

    struct NVVL_PicLayer b;
    memset(&b, 0, sizeof(b));
    b.desc.count = sequence_count;
    b.desc.channels = 3;
    b.desc.width = sequence_width;
    b.desc.height = sequence_height;
    b.type = PDT_BYTE;

    if(cudaMallocPitch(&b.data, &b.desc.stride.y, b.desc.width, b.desc.height * sequence_count * 3) != cudaSuccess) {
        fprintf(stderr, "Unable to allocate buffer in device memory\n");
        exit(1);
    }
    b.desc.stride.x = 1;
    b.desc.stride.c = b.desc.stride.y*b.desc.height;
    b.desc.stride.n = b.desc.stride.c*3;
    nvvl_set_layer(sequence, &b, "pixels");

    nvvl_read_sequence(loader, argv[1], 0, sequence_count);
    nvvl_receive_frames(loader, sequence);
    write_frames(sequence);

    nvvl_destroy_video_loader(loader);
    return 0;
}
