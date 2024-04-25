#include "Component_Pan.h"

using namespace pipeline;
const std::string Component_Pan::privateNames[1] = {"pan"};

Component_Pan::Component_Pan(const audioFormatInfo* audioInfo):AComponent(audioInfo, 1, this->privateNames, COMP_PAN){
    defaultSettings();
}

Component_Pan::~Component_Pan(){

}

void Component_Pan::apply(pipelineAudioBuffer_CUDA* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        buffer->d_bufferR[i] = pan >= 0.5 ? buffer->d_bufferR[i] : buffer->d_bufferR[i]*pan*2;
        buffer->d_bufferL[i] = pan <= 0.5 ? buffer->d_bufferL[i] : buffer->d_bufferL[i]*(2-pan*2);
    }
}

void Component_Pan::clear(){

}

void Component_Pan::defaultSettings(){
    settings.values[0] = 0.5; //pan
}
