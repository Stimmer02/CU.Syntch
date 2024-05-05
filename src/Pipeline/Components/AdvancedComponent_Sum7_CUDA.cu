#include "AdvancedComponent_Sum7_CUDA.h"

using namespace pipeline;
const std::string AdvancedComponent_Sum7_CUDA::privateNames[7] = {"vol0", "vol1", "vol2", "vol3", "vol4", "vol5", "vol6"};

__global__ void kernel_AdvancedSum7(float* bufferR, float* bufferL, float* settings, advancedComponentConnection_CUDA* connections, uint size){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    float& vol0 = settings[0];
    float& vol1 = settings[1];
    float& vol2 = settings[2];
    float& vol3 = settings[3];
    float& vol4 = settings[4];
    float& vol5 = settings[5];
    float& vol6 = settings[6];
    if (i < size){
        bufferL[i] = 
        connections[0].bufferL[i] * vol0 + 
        connections[1].bufferL[i] * vol1 + 
        connections[2].bufferL[i] * vol2 + 
        connections[3].bufferL[i] * vol3 + 
        connections[4].bufferL[i] * vol4 + 
        connections[5].bufferL[i] * vol5 + 
        connections[6].bufferL[i] * vol6;

        bufferR[i] = 
        connections[0].bufferR[i] * vol0 + 
        connections[1].bufferR[i] * vol1 + 
        connections[2].bufferR[i] * vol2 + 
        connections[3].bufferR[i] * vol3 + 
        connections[4].bufferR[i] * vol4 + 
        connections[5].bufferR[i] * vol5 + 
        connections[6].bufferR[i] * vol6;
    }
}

AdvancedComponent_Sum7_CUDA::AdvancedComponent_Sum7_CUDA(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer): AAdvancedComponent_CUDA(audioInfo, 7, privateNames, 7, boundBuffer){
    defaultSettings();
}

AdvancedComponent_Sum7_CUDA::~AdvancedComponent_Sum7_CUDA(){}

void AdvancedComponent_Sum7_CUDA::apply(pipelineAudioBuffer_CUDA* buffer){
    uint blockCount = (buffer->size + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    kernel_AdvancedSum7<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, settings.d_values, d_connections, buffer->size);
}

void AdvancedComponent_Sum7_CUDA::clear(){}

void AdvancedComponent_Sum7_CUDA::defaultSettings(){
    settings.values[0] = 0.15;//vol0
    settings.values[1] = 0.15;//vol1
    settings.values[2] = 0.15;//vol2
    settings.values[3] = 0.15;//vol3
    settings.values[4] = 0.15;//vol4
    settings.values[5] = 0.15;//vol5
    settings.values[6] = 0.15;//vol6
    settings.copyToDevice();
}

bool AdvancedComponent_Sum7_CUDA::allNeededConnections(){
    return maxConnections == connectionsCount;
}
