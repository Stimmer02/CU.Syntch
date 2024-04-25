#include "BufferConverter_Stereo8_CUDA.h"

__global__ void BufferConverter_Stereo8_CUDA::toPCM_kernel(const float* bufferL, const float* bufferR){
    const uint maxValue = 0x0000007F;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        i *= 2;
        d_buffer[i++] = maxValue * bufferL[i] + 127;
        d_buffer[i] = maxValue * bufferR[i] + 127;
    }
}
