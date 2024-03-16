#include "Component_Destroy.h"

using namespace pipeline;
const std::string Component_Destroy::privateNames[1] = {"subtract"};

Component_Destroy::Component_Destroy(const audioFormatInfo* audioInfo):AComponent(audioInfo, 1, this->privateNames, COMP_DESTROY){
    defaultSettings();
}

Component_Destroy::~Component_Destroy(){

}

void Component_Destroy::apply(pipelineAudioBuffer* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        buffer->bufferR[i] += (buffer->bufferR[i] > 0 ? -subtract : subtract) * buffer->bufferR[i] != 0;
        buffer->bufferL[i] += (buffer->bufferL[i] > 0 ? -subtract : subtract) * buffer->bufferL[i] != 0;
    }
}

void Component_Destroy::clear(){

}

void Component_Destroy::defaultSettings(){
    settings.values[0] = 0.1; //subtract
}
