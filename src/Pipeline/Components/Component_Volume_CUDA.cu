#include "Component_Volume_CUDA.h"

using namespace pipeline;
const std::string Component_Volume_CUDA::privateNames[1] = {"vol"};

__global__ void kernel_volume(float* bufferR, float* bufferL, float* settings, uint size){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    float& vol = settings[0];
    if (i < size){
        bufferR[i] *= vol;
        bufferL[i] *= vol;
    }
}

Component_Volume_CUDA::Component_Volume_CUDA(const audioFormatInfo* audioInfo):AComponent_CUDA(audioInfo, 1, this->privateNames, COMP_VOLUME){
    defaultSettings();
}

Component_Volume_CUDA::~Component_Volume_CUDA(){

}

void Component_Volume_CUDA::apply(pipelineAudioBuffer_CUDA* buffer){
    uint blockCount = (buffer->size + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    kernel_volume<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, settings.d_values, buffer->size);
}

void Component_Volume_CUDA::clear(){

}

void Component_Volume_CUDA::defaultSettings(){
    settings.values[0] = 1.0; //vol
    settings.copyToDevice();
}
