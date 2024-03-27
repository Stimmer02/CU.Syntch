#include "AdvancedComponent_Sum7.h"

using namespace pipeline;
const std::string AdvancedComponent_Sum7::privateNames[7] = {"vol0", "vol1", "vol2", "vol3", "vol4", "vol5", "vol6"};

AdvancedComponent_Sum7::AdvancedComponent_Sum7(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer): AAdvancedComponent(audioInfo, 7, privateNames, 7, boundBuffer){
    defaultSettings();
}

AdvancedComponent_Sum7::~AdvancedComponent_Sum7(){}

void AdvancedComponent_Sum7::apply(pipelineAudioBuffer* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        buffer->bufferL[i] = 
        connections[0]->buffer.bufferL[i] * vol0 + 
        connections[1]->buffer.bufferL[i] * vol1 + 
        connections[2]->buffer.bufferL[i] * vol2 + 
        connections[3]->buffer.bufferL[i] * vol3 + 
        connections[4]->buffer.bufferL[i] * vol4 + 
        connections[5]->buffer.bufferL[i] * vol5 + 
        connections[6]->buffer.bufferL[i] * vol6;

        buffer->bufferR[i] = 
        connections[0]->buffer.bufferR[i] * vol0 + 
        connections[1]->buffer.bufferR[i] * vol1 + 
        connections[2]->buffer.bufferR[i] * vol2 + 
        connections[3]->buffer.bufferR[i] * vol3 + 
        connections[4]->buffer.bufferR[i] * vol4 + 
        connections[5]->buffer.bufferR[i] * vol5 + 
        connections[6]->buffer.bufferR[i] * vol6;
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
