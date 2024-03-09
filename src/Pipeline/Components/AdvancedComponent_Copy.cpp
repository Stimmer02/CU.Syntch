#include "AdvancedComponent_Copy.h"

using namespace pipeline;
const std::string AdvancedComponent_Copy::privateNames[1] = {"vol"};

AdvancedComponent_Copy::AdvancedComponent_Copy(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer): AAdvancedComponent(audioInfo, 1, privateNames, 1, boundBuffer){
    defaultSettings();
}

AdvancedComponent_Copy::~AdvancedComponent_Copy(){}

void AdvancedComponent_Copy::apply(pipelineAudioBuffer* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        buffer->bufferL[i] = connections[0]->buffer.bufferL[i] * vol;
        buffer->bufferR[i] = connections[0]->buffer.bufferR[i] * vol;
    }
}

void AdvancedComponent_Copy::clear(){}

void AdvancedComponent_Copy::defaultSettings(){
    settings.values[0] = 1.0; //vol
}

bool AdvancedComponent_Copy::allNeededConnections(){
    return maxConnections == connectionsCount;
}
