#include "BufferConverter_Mono16_CUDA.h"

__global__ void BufferConverter_Mono16_CUDA::toPCM_kernel(const float* bufferL, const float* bufferR){
    const uint maxValue = 0x00007FFF;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        int value = maxValue * bufferL[i];
        // int value = maxValue * (bufferL[i] + bufferR[i]) / 2;
        i *= 2;
        d_buffer[i] = value;
        d_buffer[i + 1] = value >> 8;
    }
}
