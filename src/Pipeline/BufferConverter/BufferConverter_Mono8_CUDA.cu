#include "BufferConverter_Mono8_CUDA.h"

__global__ void toPCMMono8_kernel(const float* bufferL, const float* bufferR, uint8_t* output, const uint sampleSize){
    const uint maxValue = 0x00007F;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        // buffer[i] = maxValue * (bufferL[i] + bufferR[i])/2 + 127;
        output[i] = maxValue * bufferL[i] + 127;
    }
}

BufferConverter_Mono8_CUDA::BufferConverter_Mono8_CUDA(const uint& sampleSize) : sampleSize(sampleSize){
    cudaMalloc((void**)(&d_buffer), sampleSize);
}

BufferConverter_Mono8_CUDA::~BufferConverter_Mono8_CUDA(){
    cudaFree(this->d_buffer);
}

void BufferConverter_Mono8_CUDA::toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer){
    uint blockCount = (sampleSize + CUDA_BUFFER_CONVERTER_BLOCK_SIZE - 1) / CUDA_BUFFER_CONVERTER_BLOCK_SIZE;
    toPCMMono8_kernel<<<blockCount, CUDA_BUFFER_CONVERTER_BLOCK_SIZE>>>(pipelineBuffer->d_bufferL, pipelineBuffer->d_bufferR, d_buffer, sampleSize);
    cudaDeviceSynchronize();
    cudaMemcpy(pcmBuffer->buff, d_buffer, sampleSize, cudaMemcpyDeviceToHost);
}
