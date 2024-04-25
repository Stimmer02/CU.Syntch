#include "BufferConverter_Mono24_CUDA.h"

__global__ void BufferConverter_Mono24_CUDA::toPCM_kernel(const float* bufferL, const float* bufferR){
    const uint maxValue = 0x007FFFFF;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        int value = maxValue * bufferL[i];
        // int value = maxValue * (bufferL[i] + bufferR[i]) / 2;
        i *= 3;
        d_buffer[i++] = value;
        d_buffer[i++] = value >> 8;
        d_buffer[i] = value >> 16;
    }
}
