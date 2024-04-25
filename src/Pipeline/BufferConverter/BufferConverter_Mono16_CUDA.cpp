#include "BufferConverter_Mono16_CUDA.h"

BufferConverter_Mono16_CUDA::BufferConverter_Mono16_CUDA(const uint& sampleSize) : sampleSize(sampleSize){
    cudaMalloc((void**)(&d_buffer), sampleSize*2);
}

BufferConverter_Mono16_CUDA::~BufferConverter_Mono16_CUDA(){
    cudaFree(this->d_buffer);
}

void BufferConverter_Mono16_CUDA::toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer){
    uint blockCount = (sampleSize + CUDA_BUFFER_CONVERTER_BLOCK_SIZE - 1) / CUDA_BUFFER_CONVERTER_BLOCK_SIZE;
    toPCM_kernel<<<blockCount, CUDA_BUFFER_CONVERTER_BLOCK_SIZE>>>(pipelineBuffer->d_bufferL, pipelineBuffer->d_bufferR);
    cudaMemcpy(pcmBuffer->buff, d_buffer, sampleSize*2, cudaMemcpyDeviceToHost);
}
