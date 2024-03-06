#include "Component_Volume.h"

using namespace pipeline;
const std::string Component_Volume::privateNames[1] = {"vol"};

Component_Volume::Component_Volume(const audioFormatInfo* audioInfo):AComponent(audioInfo, 1, this->privateNames, COMP_VOLUME){
    defaultSettings();
}

Component_Volume::~Component_Volume(){

}

void Component_Volume::apply(pipelineAudioBuffer* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        buffer->bufferR[i] *= vol;
        buffer->bufferL[i] *= vol;
    }
}

void Component_Volume::clear(){

}

void Component_Volume::defaultSettings(){
    settings.values[0] = 1.0; //vol
}
