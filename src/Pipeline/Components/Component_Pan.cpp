#include "Component_Pan.h"

using namespace pipeline;
const std::string Component_Pan::privateNames[1] = {"pan"};

Component_Pan::Component_Pan(const audioFormatInfo* audioInfo):AComponent(audioInfo, 1, this->privateNames, COMP_PAN){
    defaultSettings();
}

Component_Pan::~Component_Pan(){

}

void Component_Pan::apply(pipelineAudioBuffer* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        buffer->bufferR[i] = pan >= 0.5 ? buffer->bufferR[i] : buffer->bufferR[i]*pan*2;
        buffer->bufferL[i] = pan <= 0.5 ? buffer->bufferL[i] : buffer->bufferL[i]*(2-pan*2);
    }
}

void Component_Pan::clear(){

}

void Component_Pan::defaultSettings(){
    settings.values[0] = 0.5; //pan
}
