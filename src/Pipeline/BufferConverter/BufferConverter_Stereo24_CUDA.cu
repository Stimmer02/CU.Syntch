#include "BufferConverter_Stereo24_CUDA.h"

__global__ void toPCMStereo24_kernel(const float* bufferL, const float* bufferR, uint8_t* output, const uint sampleSize){
    const uint maxValue = 0x007FFFFF;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        int valueL = maxValue * bufferL[i];
        int valueR = maxValue * bufferR[i];
        i *= 6;
        output[i++] = valueL;
        output[i++] = valueL >> 8;
        output[i++] = valueL >> 16;
        output[i++] = valueR;
        output[i++] = valueR >> 8;
        output[i] = valueR >> 16;
    }
}

BufferConverter_Stereo24_CUDA::BufferConverter_Stereo24_CUDA(const uint& sampleSize) : sampleSize(sampleSize){
    cudaMalloc((void**)(&d_buffer), sampleSize*6);
}

BufferConverter_Stereo24_CUDA::~BufferConverter_Stereo24_CUDA(){
    cudaFree(this->d_buffer);
}

void BufferConverter_Stereo24_CUDA::toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer){
    uint blockCount = (sampleSize + CUDA_BUFFER_CONVERTER_BLOCK_SIZE - 1) / CUDA_BUFFER_CONVERTER_BLOCK_SIZE;
    toPCMStereo24_kernel<<<blockCount, CUDA_BUFFER_CONVERTER_BLOCK_SIZE>>>(pipelineBuffer->d_bufferL, pipelineBuffer->d_bufferR, d_buffer, sampleSize);
    cudaDeviceSynchronize();
    cudaMemcpy(pcmBuffer->buff, d_buffer, sampleSize*6, cudaMemcpyDeviceToHost);
}
