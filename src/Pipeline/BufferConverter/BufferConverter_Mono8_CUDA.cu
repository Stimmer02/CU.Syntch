#include "BufferConverter_Mono8_CUDA.h"

__global__ void BufferConverter_Mono8_CUDA::toPCM_kernel(const float* bufferL, const float* bufferR){
    const uint maxValue = 0x00007F;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        // d_buffer[i] = maxValue * (bufferL[i] + bufferR[i])/2 + 127;
        d_buffer[i] = maxValue * bufferL[i] + 127;
    }
}