#include "AdvancedComponent_Sum2_CUDA.h"

using namespace pipeline;
const std::string AdvancedComponent_Sum2_CUDA::privateNames[2] = {"vol0", "vol1"};

__global__ void kernel_AdvancedSum2(float* bufferR, float* bufferL, float* settings, advancedComponentConnection_CUDA* connections, uint size){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    float& vol0 = settings[0];
    float& vol1 = settings[1];
    if (i < size){
        bufferL[i] = connections[0].bufferL[i] * vol0 + connections[1].bufferL[i] * vol1;
        bufferR[i] = connections[0].bufferR[i] * vol0 + connections[1].bufferR[i] * vol1;
    }
}

AdvancedComponent_Sum2_CUDA::AdvancedComponent_Sum2_CUDA(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer): AAdvancedComponent_CUDA(audioInfo, 2, privateNames, 2, boundBuffer){
    defaultSettings();
}

AdvancedComponent_Sum2_CUDA::~AdvancedComponent_Sum2_CUDA(){}

void AdvancedComponent_Sum2_CUDA::apply(pipelineAudioBuffer_CUDA* buffer){
    uint blockCount = (buffer->size + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    kernel_AdvancedSum2<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, settings.d_values, d_connections, buffer->size);
}

void AdvancedComponent_Sum2_CUDA::clear(){}

void AdvancedComponent_Sum2_CUDA::defaultSettings(){
    settings.values[0] = 0.5;//vol0
    settings.values[1] = 0.5;//vol1
    settings.copyToDevice();
}

bool AdvancedComponent_Sum2_CUDA::allNeededConnections(){
    return maxConnections == connectionsCount;
}
