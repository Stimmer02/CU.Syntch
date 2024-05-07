#include "Component_Pan_CUDA.h"

using namespace pipeline;
const std::string Component_Pan_CUDA::privateNames[1] = {"pan"};

__global__ void kernel_pan(float* bufferR, float* bufferL, float* settings, uint size){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    float& pan = settings[0];
    if (i < size){
        bufferR[i] = pan >= 0.5 ? bufferR[i] : bufferR[i]*pan*2;
        bufferL[i] = pan <= 0.5 ? bufferL[i] : bufferL[i]*(2-pan*2);;
    }
}

Component_Pan_CUDA::Component_Pan_CUDA(const audioFormatInfo* audioInfo):AComponent_CUDA(audioInfo, 1, this->privateNames, COMP_PAN){
    defaultSettings();
}

Component_Pan_CUDA::~Component_Pan_CUDA(){

}

void Component_Pan_CUDA::apply(pipelineAudioBuffer_CUDA* buffer){
    uint blockCount = (buffer->size + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    kernel_pan<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, settings.d_values, buffer->size);
}

void Component_Pan_CUDA::clear(){

}

void Component_Pan_CUDA::defaultSettings(){
    settings.values[0] = 0.5; //pan
    settings.copyToDevice();
}
