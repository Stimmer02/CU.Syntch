#include "BufferConverter_Mono32_CUDA.h"

__global__ void toPCMMono32_kernel(const float* bufferL, const float* bufferR, uint8_t* output, const uint sampleSize){
    const uint maxValue = 0x7FFFFF80;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        int value = maxValue * bufferL[i];
        // int value = maxValue * (bufferL[i] + bufferR[i]) / 2;
        i *= 4;
        output[i++] = value;
        output[i++] = value >> 8;
        output[i++] = value >> 16;
        output[i] = value >> 24;
    }
}

BufferConverter_Mono32_CUDA::BufferConverter_Mono32_CUDA(const uint& sampleSize) : sampleSize(sampleSize){
    cudaMalloc((void**)(&d_buffer), sampleSize*4);
}

BufferConverter_Mono32_CUDA::~BufferConverter_Mono32_CUDA(){
    cudaFree(this->d_buffer);
}

void BufferConverter_Mono32_CUDA::toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer){
    uint blockCount = (sampleSize + CUDA_BUFFER_CONVERTER_BLOCK_SIZE - 1) / CUDA_BUFFER_CONVERTER_BLOCK_SIZE;
    toPCMMono32_kernel<<<blockCount, CUDA_BUFFER_CONVERTER_BLOCK_SIZE>>>(pipelineBuffer->d_bufferL, pipelineBuffer->d_bufferR, d_buffer, sampleSize);
    cudaDeviceSynchronize();
    cudaMemcpy(pcmBuffer->buff, d_buffer, sampleSize*4, cudaMemcpyDeviceToHost);
}
