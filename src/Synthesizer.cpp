#include "Synthesizer.h"
#include "Synthesizer/IGenerator.h"

using namespace synthesizer;

Synthesizer::Synthesizer(const audioFormatInfo& audioInfo, const ushort& keyCount){
    settings.keyCount = keyCount;
    settings.sampleSize = audioInfo.sampleSize;
    settings.sampleRate = audioInfo.sampleRate;
    settings.pitch = -24;
    settings.volume = 0.15;
    settings.attack.set(0.2, audioInfo.sampleRate);
    settings.sustain.set(0, audioInfo.sampleRate);
    settings.fade.set(0, audioInfo.sampleRate);
    settings.release.set(0.2, audioInfo.sampleRate);
    settings.stereoMix = 0.5;

    settings.maxValue = 0;
    uint a = 1;
    for (uint i = 1; i < audioInfo.bitDepth; i++){
        settings.maxValue += a;
        a <<= 1;
    }
    settings.maxValue -= 1;

    soundGenerator = new Generator_Sine();
    generatorType = SINE;

    notes = new noteBuffer[keyCount];
    for (uint i = 0; i < keyCount; i++){
        notes[i].init(audioInfo.sampleSize);
    }

    calculateFrequencies();
    calculateStereoFactor();
}

Synthesizer::~Synthesizer(){
    delete soundGenerator;
    delete[] notes;
}

struct settings* Synthesizer::getSettings(){
    return &this->settings;
}

generator_type Synthesizer::getGeneratorType(){
    return generatorType;
}

void Synthesizer::setGenerator(generator_type type){
    if (generatorType == type){
        return;
    }

    generatorType = type;
    delete soundGenerator;
    switch (type) {
        case SINE:
            soundGenerator = new Generator_Sine();
            break;
        case SQUARE:
            soundGenerator = new Generator_Square();
            break;
        case SAWTOOTH:
            soundGenerator = new Generator_Sawtooth();
            break;
    }
    calculateFrequencies();
}

char Synthesizer::setPitch(const char& value, const bool& add){
    if (add){
        settings.pitch += value;
    } else {
        settings.pitch = value;
    }

    calculateFrequencies();
    return settings.pitch;
}

void Synthesizer::calculateFrequencies(){
    if (generatorType == SINE){
        for (int i = 0; i < settings.keyCount; i++){
            notes[i].frequency = 440.0 * pow(2.0, (i+settings.pitch)/12.0);
            notes[i].multiplier = PI*2 * notes[i].frequency / settings.sampleRate;
        }
    } else {
        for (int i = 0; i < settings.keyCount; i++){
            notes[i].frequency = 440.0 * pow(2.0, (i+settings.pitch)/12.0);
            notes[i].multiplier = settings.sampleRate / notes[i].frequency;
        }
    }
}

void Synthesizer::calculateStereoFactor(){
    double stereoFactorL = 1, stereoFactorR = settings.stereoMix;
    double multiplierStep = (stereoFactorL-stereoFactorR) / settings.keyCount;

    for (uint i = 0; i < settings.keyCount; i++){
        notes[i].stereoFactorL = stereoFactorL;
        notes[i].stereoFactorR = stereoFactorR;
        stereoFactorL -= multiplierStep;
        stereoFactorR += multiplierStep;
    }
}

void Synthesizer::mixAudio(pipelineAudioBuffer*& audioBuffer){

    for (uint i = 0; i < settings.sampleSize; i++){
        audioBuffer->bufferL[i] = notes[0].buffer[i] * notes[0].stereoFactorL;
        audioBuffer->bufferR[i] = notes[0].buffer[i] * notes[0].stereoFactorR;
    }
    for (uint i = 1; i < settings.keyCount; i++){
        for (uint j = 0; j < settings.sampleSize; j++){
            audioBuffer->bufferL[j] += notes[i].buffer[j] * notes[i].stereoFactorL;
            audioBuffer->bufferR[j] += notes[i].buffer[j] * notes[i].stereoFactorR;
        }
    }
}

void Synthesizer::generateSample(pipelineAudioBuffer* audioBuffer,  const keyboardTransferBuffer* keyboardState){
    for (uint i = 0; i < settings.keyCount; i++){
        soundGenerator->generate(notes[i], keyboardState->buffer[i], settings);
    }
    mixAudio(audioBuffer);
}
