#include "BufferConverter_Stereo32_CUDA.h"

__global__ void BufferConverter_Stereo32_CUDA::toPCM_kernel(const float* bufferL, const float* bufferR){
    const uint maxValue = 0x7FFFFF80;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        int valueL = maxValue * bufferL[i];
        int valueR = maxValue * bufferR[i];
        i *= 8;
        d_buffer[i++] = valueL;
        d_buffer[i++] = valueL >> 8;
        d_buffer[i++] = valueL >> 16;
        d_buffer[i++] = valueL >> 24;
        d_buffer[i++] = valueR;
        d_buffer[i++] = valueR >> 8;
        d_buffer[i++] = valueR >> 16;
        d_buffer[i] = valueR >> 24;
    }
}
