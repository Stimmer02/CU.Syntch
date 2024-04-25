#include "Component_Distortion.h"

using namespace pipeline;
const std::string Component_Distortion::privateNames[4] = {"gain", "compress", "symmetry", "vol"};

Component_Distortion::Component_Distortion(const audioFormatInfo* audioInfo):AComponent(audioInfo, 4, this->privateNames, COMP_DISTORION){
    defaultSettings();
}

Component_Distortion::~Component_Distortion(){

}

void Component_Distortion::apply(pipelineAudioBuffer_CUDA* buffer){
    float positiveGain = gain * (1 + symmetry);
    float negativeGain = gain * (1 - symmetry);
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        if (buffer->d_bufferR[i] > 0){
            buffer->d_bufferR[i] *= positiveGain;
            if (buffer->d_bufferR[i] > compress){
                buffer->d_bufferR[i] = compress;
            }
        } else {
            buffer->d_bufferR[i] *= negativeGain;
            if (buffer->d_bufferR[i] < -compress){
                buffer->d_bufferR[i] = -compress;
            }
        }


        if (buffer->d_bufferL[i] > 0){
            buffer->d_bufferL[i] *= positiveGain;
            if (buffer->d_bufferL[i] > compress){
                buffer->d_bufferL[i] = compress;
            }
        } else {
            buffer->d_bufferL[i] *= negativeGain;
            if (buffer->d_bufferL[i] < -compress){
                buffer->d_bufferL[i] = -compress;
            }
        }
    }
}

void Component_Distortion::clear(){

}

void Component_Distortion::defaultSettings(){
    settings.values[0] = 4.0;  //gain
    settings.values[1] = 0.5;  //compress
    settings.values[2] = 0.17; //symmetry
    settings.values[3] = 1.0;  //vol
}
