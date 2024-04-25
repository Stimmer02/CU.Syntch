#include "BufferConverter_Mono32_CUDA.h"

BufferConverter_Mono32_CUDA::BufferConverter_Mono32_CUDA(const uint& sampleSize) : sampleSize(sampleSize){
    cudaMalloc((void**)(&d_buffer), sampleSize*4);
}

BufferConverter_Mono32_CUDA::~BufferConverter_Mono32_CUDA(){
    cudaFree(this->d_buffer);
}

void BufferConverter_Mono32_CUDA::toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer){
    uint blockCount = (sampleSize + CUDA_BUFFER_CONVERTER_BLOCK_SIZE - 1) / CUDA_BUFFER_CONVERTER_BLOCK_SIZE;
    toPCM_kernel<<<blockCount, CUDA_BUFFER_CONVERTER_BLOCK_SIZE>>>(pipelineBuffer->d_bufferL, pipelineBuffer->d_bufferR);
    cudaMemcpy(pcmBuffer->buff, d_buffer, sampleSize*4, cudaMemcpyDeviceToHost);
}
