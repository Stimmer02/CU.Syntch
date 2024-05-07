#ifndef PIPELINEAUDIOBUFFER_CUDA_H
#define PIPELINEAUDIOBUFFER_CUDA_H

#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cuda.h>

typedef unsigned int uint;

struct pipelineAudioBuffer_CUDA {
    float* d_bufferL;
    float* d_bufferR;
    const uint size;

    pipelineAudioBuffer_CUDA(const uint& sampleSize) : size(sampleSize){
        cudaMalloc((void**)(&d_bufferL), sampleSize * sizeof(float));
        cudaMalloc((void**)(&d_bufferR), sampleSize * sizeof(float));
    }
    ~pipelineAudioBuffer_CUDA(){
        cudaFree(d_bufferL);
        cudaFree(d_bufferR);
    }
};

#endif
