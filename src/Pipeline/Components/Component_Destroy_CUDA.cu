#include "Component_Destroy_CUDA.h"

using namespace pipeline;
const std::string Component_Destroy_CUDA::privateNames[1] = {"subtract"};

__global__ void kernel_destroy(float* bufferR, float* bufferL, float* settings, uint size){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    float& subtract = settings[0];
    if (i < size){
        bufferR[i] += (bufferR[i] > 0 ? -subtract : subtract) * bufferR[i] != 0;
        bufferL[i] += (bufferL[i] > 0 ? -subtract : subtract) * bufferL[i] != 0;
    }
}

Component_Destroy_CUDA::Component_Destroy_CUDA(const audioFormatInfo* audioInfo):AComponent_CUDA(audioInfo, 1, this->privateNames, COMP_DESTROY){
    defaultSettings();
}

Component_Destroy_CUDA::~Component_Destroy_CUDA(){

}

void Component_Destroy_CUDA::apply(pipelineAudioBuffer_CUDA* buffer){
    uint blockCount = (buffer->size + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    kernel_destroy<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, settings.d_values, buffer->size);
}

void Component_Destroy_CUDA::clear(){

}

void Component_Destroy_CUDA::defaultSettings(){
    settings.values[0] = 0.1; //subtract
    settings.copyToDevice();
}
