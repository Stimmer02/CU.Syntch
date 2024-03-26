#include "Component_Compressor.h"

using namespace pipeline;
const std::string Component_Compressor::privateNames[6] = {"threshold", "ratio", "step", "attack", "release", "vol"};

Component_Compressor::Component_Compressor(const audioFormatInfo* audioInfo):AComponent(audioInfo, 6, this->privateNames, COMP_COMPRESSOR){
    defaultSettings();
    clear();
}

Component_Compressor::~Component_Compressor(){

}

void Component_Compressor::apply(pipelineAudioBuffer* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        float rAbs = std::abs(buffer->bufferR[i]);
        float lAbs = std::abs(buffer->bufferL[i]);

        if (rAbs >= rLevel){
            rLevel += rAbs / levelRiseTime;
        } else {
            rLevel -= rAbs / levelDecreaseTime;
        }

        if (lAbs >= lLevel){
            lLevel += lAbs / levelRiseTime;
        } else {
            lLevel -= lAbs / levelDecreaseTime;
        }

        float rDiff = rLevel - threshold;
        float lDiff = lLevel - threshold;

        if (rDiff > 0){
            buffer->bufferR[i] *= (threshold + (rDiff / (step*ratio))) / rLevel;
        }

        if (lDiff > 0){
            buffer->bufferL[i] *= (threshold + (lDiff / (step*ratio))) / lLevel;
        }
        buffer->bufferR[i] *= vol;
        buffer->bufferL[i] *= vol;
    }
}

void Component_Compressor::clear(){
    lLevel = 0.0;
    rLevel = 0.0;
}

void Component_Compressor::defaultSettings(){
    settings.values[0] = 0.8;     //threshold
    settings.values[1] = 500;     //ratio
    settings.values[2] = 0.1;     //step
    settings.values[3] = 0.00001; //attack
    settings.values[4] = 0.1;     //release
    settings.values[5] = 1.0;     //vol

    levelRiseTime = audioInfo->sampleRate * attack;
    levelDecreaseTime= audioInfo->sampleRate * release;
}

void Component_Compressor::set(uint index, float value){
    switch (index) {
        case 0:
            if (value < 0){
                value = 0;
            }
            break;
        case 2:
            if (value == 0){
                value = 0.000000001;
            }
            break;
        case 3:
            if (value <= 0){
                value = 0.000000001;
            }
            levelRiseTime = audioInfo->sampleRate * value;
            break;
        case 4:
            if (value <= 0){
                value = 0.000000001;
            }
            levelDecreaseTime= audioInfo->sampleRate * value;
            break;
    }

    settings.values[index] = value;
}
