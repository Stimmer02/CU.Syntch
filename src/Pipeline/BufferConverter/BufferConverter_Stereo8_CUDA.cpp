#include "BufferConverter_Stereo8_CUDA.h"

BufferConverter_Stereo8_CUDA::BufferConverter_Stereo8_CUDA(const uint& sampleSize) : sampleSize(sampleSize){
    cudaMalloc((void**)(&d_buffer), sampleSize*2);
}

BufferConverter_Stereo8_CUDA::~BufferConverter_Stereo8_CUDA(){
    cudaFree(this->d_buffer);
}

void BufferConverter_Stereo8_CUDA::toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer){
    uint blockCount = (sampleSize + CUDA_BUFFER_CONVERTER_BLOCK_SIZE - 1) / CUDA_BUFFER_CONVERTER_BLOCK_SIZE;
    toPCM_kernel<<<blockCount, CUDA_BUFFER_CONVERTER_BLOCK_SIZE>>>(pipelineBuffer->d_bufferL, pipelineBuffer->d_bufferR);
    cudaMemcpy(pcmBuffer->buff, d_buffer, sampleSize*2, cudaMemcpyDeviceToHost);
}
