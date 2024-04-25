#include "AdvancedComponent_Sum7.h"

using namespace pipeline;
const std::string AdvancedComponent_Sum7::privateNames[7] = {"vol0", "vol1", "vol2", "vol3", "vol4", "vol5", "vol6"};

AdvancedComponent_Sum7::AdvancedComponent_Sum7(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer): AAdvancedComponent(audioInfo, 7, privateNames, 7, boundBuffer){
    defaultSettings();
}

AdvancedComponent_Sum7::~AdvancedComponent_Sum7(){}

void AdvancedComponent_Sum7::apply(pipelineAudioBuffer_CUDA* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        buffer->d_bufferL[i] = 
        connections[0]->buffer.d_bufferL[i] * vol0 + 
        connections[1]->buffer.d_bufferL[i] * vol1 + 
        connections[2]->buffer.d_bufferL[i] * vol2 + 
        connections[3]->buffer.d_bufferL[i] * vol3 + 
        connections[4]->buffer.d_bufferL[i] * vol4 + 
        connections[5]->buffer.d_bufferL[i] * vol5 + 
        connections[6]->buffer.d_bufferL[i] * vol6;

        buffer->d_bufferR[i] = 
        connections[0]->buffer.d_bufferR[i] * vol0 + 
        connections[1]->buffer.d_bufferR[i] * vol1 + 
        connections[2]->buffer.d_bufferR[i] * vol2 + 
        connections[3]->buffer.d_bufferR[i] * vol3 + 
        connections[4]->buffer.d_bufferR[i] * vol4 + 
        connections[5]->buffer.d_bufferR[i] * vol5 + 
        connections[6]->buffer.d_bufferR[i] * vol6;
    }
}

void AdvancedComponent_Sum7::clear(){}

void AdvancedComponent_Sum7::defaultSettings(){
    settings.values[0] = 0.15;//vol0
    settings.values[1] = 0.15;//vol1
    settings.values[2] = 0.15;//vol2
    settings.values[3] = 0.15;//vol3
    settings.values[4] = 0.15;//vol4
    settings.values[5] = 0.15;//vol5
    settings.values[6] = 0.15;//vol6
}

bool AdvancedComponent_Sum7::allNeededConnections(){
    return maxConnections == connectionsCount;
}
