#include "BufferConverter_Stereo16_CUDA.h"

__global__ void toPCMStereo16_kernel(const float* bufferL, const float* bufferR, uint8_t* output, const uint sampleSize){
    const uint maxValue = 0x00007FFF;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sampleSize){
        int valueL = maxValue * bufferL[i];
        int valueR = maxValue * bufferR[i];
        i *= 4;
        output[i++] = valueL;
        output[i++] = valueL >> 8;
        output[i++] = valueR;
        output[i] = valueR >> 8;
    }
}

BufferConverter_Stereo16_CUDA::BufferConverter_Stereo16_CUDA(const uint& sampleSize) : sampleSize(sampleSize){
    cudaMalloc((void**)(&d_buffer), sampleSize*4);
}

BufferConverter_Stereo16_CUDA::~BufferConverter_Stereo16_CUDA(){
    cudaFree(this->d_buffer);
}

void BufferConverter_Stereo16_CUDA::toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer){
    uint blockCount = (sampleSize + CUDA_BUFFER_CONVERTER_BLOCK_SIZE - 1) / CUDA_BUFFER_CONVERTER_BLOCK_SIZE;
    toPCMStereo16_kernel<<<blockCount, CUDA_BUFFER_CONVERTER_BLOCK_SIZE>>>(pipelineBuffer->d_bufferL, pipelineBuffer->d_bufferR, d_buffer, sampleSize);
    cudaDeviceSynchronize();
    cudaMemcpy(pcmBuffer->buff, d_buffer, sampleSize*4, cudaMemcpyDeviceToHost);
}

