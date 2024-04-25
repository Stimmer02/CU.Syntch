#include "Component_Destroy.h"

using namespace pipeline;
const std::string Component_Destroy::privateNames[1] = {"subtract"};

Component_Destroy::Component_Destroy(const audioFormatInfo* audioInfo):AComponent(audioInfo, 1, this->privateNames, COMP_DESTROY){
    defaultSettings();
}

Component_Destroy::~Component_Destroy(){

}

void Component_Destroy::apply(pipelineAudioBuffer_CUDA* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        buffer->d_bufferR[i] += (buffer->d_bufferR[i] > 0 ? -subtract : subtract) * buffer->d_bufferR[i] != 0;
        buffer->d_bufferL[i] += (buffer->d_bufferL[i] > 0 ? -subtract : subtract) * buffer->d_bufferL[i] != 0;
    }
}

void Component_Destroy::clear(){

}

void Component_Destroy::defaultSettings(){
    settings.values[0] = 0.1; //subtract
}
