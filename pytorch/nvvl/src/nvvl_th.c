#include <THC/THC.h>
#include "VideoLoader.h"

extern THCState *state;

void nvvl_sequence_stream_wait_th(PictureSequenceHandle sequence) {
    nvvl_sequence_stream_wait(sequence, THCState_getCurrentStream(state));
}
