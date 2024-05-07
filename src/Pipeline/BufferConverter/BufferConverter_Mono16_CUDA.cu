#include "BufferConverter_Mono16_CUDA.h"

__global__ void toPCMMono16_kernel(const float* bufferL, const float* bufferR, uint8_t* output, const uint sampleSize){
    const uint maxValue = 0x00007FFF;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        int value = maxValue * bufferL[i];
        // int value = maxValue * (bufferL[i] + bufferR[i]) / 2;
        i *= 2;
        output[i] = value;
        output[i + 1] = value >> 8;
    }
}

BufferConverter_Mono16_CUDA::BufferConverter_Mono16_CUDA(const uint& sampleSize) : sampleSize(sampleSize){
    cudaMalloc((void**)(&d_buffer), sampleSize*2);
}

BufferConverter_Mono16_CUDA::~BufferConverter_Mono16_CUDA(){
    cudaFree(this->d_buffer);
}

void BufferConverter_Mono16_CUDA::toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer){
    uint blockCount = (sampleSize + CUDA_BUFFER_CONVERTER_BLOCK_SIZE - 1) / CUDA_BUFFER_CONVERTER_BLOCK_SIZE;
    toPCMMono16_kernel<<<blockCount, CUDA_BUFFER_CONVERTER_BLOCK_SIZE>>>(pipelineBuffer->d_bufferL, pipelineBuffer->d_bufferR, d_buffer, sampleSize);
    cudaDeviceSynchronize();
    cudaMemcpy(pcmBuffer->buff, d_buffer, sampleSize*2, cudaMemcpyDeviceToHost);
}

