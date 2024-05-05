#include "AdvancedComponent_Copy_CUDA.h"

using namespace pipeline;
const std::string AdvancedComponent_Copy_CUDA::privateNames[1] = {"vol"};

__global__ void kernel_AdvancedCopy(float* bufferR, float* bufferL, float* settings, advancedComponentConnection_CUDA* connections, uint size){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    float& vol = settings[0];
    if (i < size){
        bufferL[i] = connections[0].bufferL[i] * vol;
        bufferR[i] = connections[0].bufferR[i] * vol;
    }
}

AdvancedComponent_Copy_CUDA::AdvancedComponent_Copy_CUDA(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer): AAdvancedComponent_CUDA(audioInfo, 1, privateNames, 1, boundBuffer){
    defaultSettings();
}

AdvancedComponent_Copy_CUDA::~AdvancedComponent_Copy_CUDA(){}

void AdvancedComponent_Copy_CUDA::apply(pipelineAudioBuffer_CUDA* buffer){
    uint blockCount = (buffer->size + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    kernel_AdvancedCopy<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, settings.d_values, d_connections, buffer->size);
}

void AdvancedComponent_Copy_CUDA::clear(){}

void AdvancedComponent_Copy_CUDA::defaultSettings(){
    settings.values[0] = 1.0; //vol
    settings.copyToDevice();
}

bool AdvancedComponent_Copy_CUDA::allNeededConnections(){
    return maxConnections == connectionsCount;
}
