#include "AdvancedComponent_Copy.h"

using namespace pipeline;
const std::string AdvancedComponent_Copy::privateNames[1] = {"vol"};

AdvancedComponent_Copy::AdvancedComponent_Copy(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer): AAdvancedComponent(audioInfo, 1, privateNames, 1, boundBuffer){
    defaultSettings();
}

AdvancedComponent_Copy::~AdvancedComponent_Copy(){}

void AdvancedComponent_Copy::apply(pipelineAudioBuffer_CUDA* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        buffer->d_bufferL[i] = connections[0]->buffer.d_bufferL[i] * vol;
        buffer->d_bufferR[i] = connections[0]->buffer.d_bufferR[i] * vol;
    }
}

void AdvancedComponent_Copy::clear(){}

void AdvancedComponent_Copy::defaultSettings(){
    settings.values[0] = 1.0; //vol
}

bool AdvancedComponent_Copy::allNeededConnections(){
    return maxConnections == connectionsCount;
}
