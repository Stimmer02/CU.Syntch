#include "Component_Echo.h"

using namespace pipeline;
const std::string Component_Echo::privateNames[5] = {"lvol", "rvol", "delay", "fade", "repeats"};

Component_Echo::Component_Echo(const audioFormatInfo* audioInfo):AComponent(audioInfo, 5, this->privateNames, COMP_ECHO), maxDelayTime(10){
    currentSample = 0;
    sampleCount = audioInfo->sampleRate * maxDelayTime / audioInfo->sampleSize;
    sampleCount += 1 ? 0 : (audioInfo->sampleRate * maxDelayTime % audioInfo->sampleSize) > 0;

    lMemory = new float*[sampleCount];
    rMemory = new float*[sampleCount];

    for (int i = 0; i < sampleCount; i++){
        lMemory[i] = new float[audioInfo->sampleSize];
        rMemory[i] = new float[audioInfo->sampleSize];
    }

    defaultSettings();
    clear();
}

Component_Echo::~Component_Echo(){
    for (int i = 0; i < sampleCount; i++){
        delete[] lMemory[i];
        delete[] rMemory[i];
    }
    delete[] lMemory;
    delete[] rMemory;
}

void Component_Echo::apply(pipelineAudioBuffer_CUDA* buffer){
    for (uint i = 0; i < audioInfo->sampleSize; i++){
        rMemory[currentSample][i] = buffer->d_bufferR[i];
        lMemory[currentSample][i] = buffer->d_bufferL[i];
    }

    uint allSamplesShift = audioInfo->sampleRate * delay;
    int sampleIndex = currentSample - allSamplesShift / audioInfo->sampleSize - 1;
    if (sampleIndex < 0){
        sampleIndex = sampleCount + sampleIndex;
    }
    uint singleSampleShift = allSamplesShift % audioInfo->sampleSize;
    uint indexShift = audioInfo->sampleSize - singleSampleShift;

    float rVolume = rvol;
    float lVolume = lvol;

    uint i = 0;
    for (; i < singleSampleShift; i++){
        buffer->d_bufferR[i] = buffer->d_bufferR[i] + rMemory[sampleIndex][i+indexShift] * rVolume;
        buffer->d_bufferL[i] = buffer->d_bufferL[i] + lMemory[sampleIndex][i+indexShift] * lVolume;
    }

    sampleIndex++;
    if (sampleIndex == sampleCount){
        sampleIndex = 0;
    }

    for (; i < audioInfo->sampleSize; i++){
        buffer->d_bufferR[i] = buffer->d_bufferR[i] + rMemory[sampleIndex][i-singleSampleShift] * rVolume;
        buffer->d_bufferL[i] = buffer->d_bufferL[i] + lMemory[sampleIndex][i-singleSampleShift] * lVolume;
    }

    for (uint repeat = 1; repeat < repeats; repeat++){
        allSamplesShift = audioInfo->sampleRate * delay * repeat;
        sampleIndex = currentSample - allSamplesShift / audioInfo->sampleSize - 1;
        if (sampleIndex < 0){
            sampleIndex = sampleCount + sampleIndex;
        }
        singleSampleShift = allSamplesShift % audioInfo->sampleSize;
        indexShift = audioInfo->sampleSize - singleSampleShift;

        rVolume *= fade;
        lVolume *= fade;

        i = 0;
        for (; i < singleSampleShift; i++){
            buffer->d_bufferR[i] += rMemory[sampleIndex][i+indexShift] * rVolume;
            buffer->d_bufferL[i] += lMemory[sampleIndex][i+indexShift] * lVolume;
        }

        sampleIndex++;
        if (sampleIndex == sampleCount){
            sampleIndex = 0;
        }

        for (; i < audioInfo->sampleSize; i++){
            buffer->d_bufferR[i] += rMemory[sampleIndex][i-singleSampleShift] * rVolume;
            buffer->d_bufferL[i] += lMemory[sampleIndex][i-singleSampleShift] * lVolume;
        }
    }



    currentSample++;
    if (currentSample == sampleCount){
        currentSample = 0;
    }
}

void Component_Echo::clear(){
    for (int i = 0; i < sampleCount; i++){
        std::memset(lMemory[i], 0, sizeof(float)*audioInfo->sampleSize);
        std::memset(rMemory[i], 0, sizeof(float)*audioInfo->sampleSize);
    }
}

void Component_Echo::defaultSettings(){
    settings.values[0] = 0.5;  //lvol
    settings.values[1] = 0.5;  //rvol
    settings.values[2] = 0.2;  //delay
    settings.values[3] = 0.7;  //fade
    settings.values[4] = 5;    //repeats
}

void Component_Echo::set(uint index, float value){
    switch (index) {
        case 2:
            if (value > maxDelayTime){
                value = maxDelayTime;
            } else if (value < 0){
                value = 0;
            }
        case 4:
            if (value * delay > maxDelayTime){
                value = maxDelayTime / delay;
            } else if (value < 1){
                value = 1;
            }
    }

    settings.values[index] = value;
}