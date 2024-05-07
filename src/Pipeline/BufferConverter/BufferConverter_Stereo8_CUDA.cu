#include "BufferConverter_Stereo8_CUDA.h"

__global__ void toPCMStereo8_kernel(const float* bufferL, const float* bufferR, uint8_t* output, const uint sampleSize){
    const uint maxValue = 0x0000007F;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        i *= 2;
        output[i++] = maxValue * bufferL[i] + 127;
        output[i] = maxValue * bufferR[i] + 127;
    }
}

BufferConverter_Stereo8_CUDA::BufferConverter_Stereo8_CUDA(const uint& sampleSize) : sampleSize(sampleSize){
    cudaMalloc((void**)(&d_buffer), sampleSize*2);
}

BufferConverter_Stereo8_CUDA::~BufferConverter_Stereo8_CUDA(){
    cudaFree(this->d_buffer);
}

void BufferConverter_Stereo8_CUDA::toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer){
    uint blockCount = (sampleSize + CUDA_BUFFER_CONVERTER_BLOCK_SIZE - 1) / CUDA_BUFFER_CONVERTER_BLOCK_SIZE;
    toPCMStereo8_kernel<<<blockCount, CUDA_BUFFER_CONVERTER_BLOCK_SIZE>>>(pipelineBuffer->d_bufferL, pipelineBuffer->d_bufferR, d_buffer, sampleSize);
    cudaDeviceSynchronize();
    cudaMemcpy(pcmBuffer->buff, d_buffer, sampleSize*2, cudaMemcpyDeviceToHost);
}
