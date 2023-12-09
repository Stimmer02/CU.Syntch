#include "Synthesizer.h"
#include "Synthesizer/AGenerator.h"

using namespace synthesizer;

Synthesizer::Synthesizer(const audioFormatInfo& audioInfo, const ushort& keyCount){
    settings.keyCount = keyCount;
    settings.sampleSize = audioInfo.sampleSize;
    settings.sampleRate = audioInfo.sampleRate;
    settings.pitch = 0;
    settings.volume = 0.15;
    settings.attack.set(1, audioInfo.sampleRate);
    settings.sustain.set(1, audioInfo.sampleRate);
    settings.fade.set(0.5, audioInfo.sampleRate);
    settings.rawFadeTo = 0.8;
    settings.fadeTo = 0.8;
    settings.release.set(1.2, audioInfo.sampleRate);
    settings.stereoMix = 0.5;

    settings.maxValue = 0;
    uint a = 1;
    for (uint i = 1; i < audioInfo.bitDepth; i++){
        settings.maxValue += a;
        a <<= 1;
    }
    settings.maxValue -= 1;

    soundGenerator = new Generator_Sine();
    generatorType = generator_type(0);

    notes = new noteBuffer[keyCount];
    for (uint i = 0; i < keyCount; i++){
        notes[i].init(audioInfo.sampleSize);
    }

    calculateFrequencies();
    calculateStereoFactor();
    dynamicsController.calculateDynamicsProfile(settings);
    dynamicsController.calculateReleaseProfile(settings);
}

Synthesizer::~Synthesizer(){
    delete soundGenerator;
    delete[] notes;
}

const struct settings* Synthesizer::getSettings(){
    return &this->settings;
}

void Synthesizer::setSettings(const settings_name& settingsName, const float& value){
    switch (settingsName) {
        case synthesizer::PITCH:
            settings.pitch = value;
            calculateFrequencies();
            break;

        case synthesizer::ATTACK:
            settings.attack.set(value, settings.sampleRate);
            dynamicsController.calculateDynamicsProfile(settings);
            break;

        case synthesizer::SUSTAIN:
            settings.sustain.set(value, settings.sampleRate);
            dynamicsController.calculateDynamicsProfile(settings);
            break;

        case synthesizer::FADE:
            settings.fade.set(value, settings.sampleRate);
            dynamicsController.calculateDynamicsProfile(settings);
            break;

        case synthesizer::RELEASE:
            settings.release.set(value, settings.sampleRate);
            dynamicsController.calculateReleaseProfile(settings);
            break;

        case synthesizer::VOLUME:
            settings.volume = value;
            break;

        case synthesizer::STEREO:
            settings.stereoMix = value;
            calculateStereoFactor();
            break;

        case synthesizer::FADETO:
            settings.rawFadeTo = value;
            if (settings.fade.duration != 0){
                settings.fadeTo = value;
            }
            dynamicsController.calculateDynamicsProfile(settings);
            break;
    }
}

generator_type Synthesizer::getGeneratorType(){
    return generatorType;
}

void Synthesizer::setGenerator(const generator_type& type){
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
        case TRIANGLE:
            soundGenerator = new Generator_Triangle();
            break;
        case NOISE1:
            soundGenerator = new Generator_Noise1();
            break;
        }
    calculateFrequencies();
}


void Synthesizer::calculateFrequencies(){
    switch (generatorType) {
        case SINE:
            for (int i = 0; i < settings.keyCount; i++){
                notes[i].frequency = 440.0 * pow(2.0, (i+settings.pitch-69)/12.0);
                notes[i].multiplier = PI*2 * notes[i].frequency / settings.sampleRate;
            }
            break;
        case SQUARE:
        case SAWTOOTH:
            for (int i = 0; i < settings.keyCount; i++){
                notes[i].frequency = 440.0 * pow(2.0, (i+settings.pitch-69)/12.0);
                notes[i].multiplier = settings.sampleRate / notes[i].frequency;
            }
            break;
        case TRIANGLE:
            for (int i = 0; i < settings.keyCount; i++){
                notes[i].frequency = 440.0 * pow(2.0, (i+settings.pitch-69)/12.0);
                notes[i].multiplier = settings.sampleRate / notes[i].frequency / 2;
            }
            break;
        case NOISE1:
            for (int i = 0; i < settings.keyCount; i++){
                notes[i].frequency = 440.0 * pow(2.0, (i+settings.pitch-69)/12.0);
                notes[i].multiplier = settings.sampleRate / notes[i].frequency;
            }
            break;
        }
}

void Synthesizer::calculateStereoFactor(){
    float stereoFactorL = 1, stereoFactorR = settings.stereoMix;
    float multiplierStep = (stereoFactorL-stereoFactorR) / settings.keyCount;

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
        soundGenerator->generate(notes[i], keyboardState->buffer[i], settings, dynamicsController.getDynamicsProfile(), dynamicsController.getReleaseProfile());
    }
    mixAudio(audioBuffer);
}

char Synthesizer::saveConfig(std::string path){
    std::ofstream file(path, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    if (file.fail()){
        std::fprintf(stderr, "ERR Synthesizer::saveConfig: CANNOT OPEN FILE %s\n", path.c_str());
        return 1;
    }
    file.write(reinterpret_cast<char*>(&settings.pitch), sizeof(settings.pitch));
    file.write(reinterpret_cast<char*>(&settings.volume), sizeof(settings.volume));
    file.write(reinterpret_cast<char*>(&settings.attack.raw), sizeof(settings.attack.raw));
    file.write(reinterpret_cast<char*>(&settings.sustain.raw), sizeof(settings.sustain.raw));
    file.write(reinterpret_cast<char*>(&settings.fade.raw), sizeof(settings.fade.raw));
    file.write(reinterpret_cast<char*>(&settings.fadeTo), sizeof(settings.fadeTo));
    file.write(reinterpret_cast<char*>(&settings.rawFadeTo), sizeof(settings.rawFadeTo));
    file.write(reinterpret_cast<char*>(&settings.release.raw), sizeof(settings.release.raw));
    file.write(reinterpret_cast<char*>(&settings.dynamicsDuration), sizeof(settings.dynamicsDuration));
    file.write(reinterpret_cast<char*>(&settings.stereoMix), sizeof(settings.stereoMix));
    file.write(reinterpret_cast<char*>(&generatorType), sizeof(generatorType));


    file.close();
    return 0;
}

char Synthesizer::loadConfig(std::string path){
    std::ifstream file(path, std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
    if (file.fail()){
        std::fprintf(stderr, "ERR Synthesizer::loadConfig: CANNOT OPEN FILE %s\n", path.c_str());
        return 1;
    }
    long size = file.tellg();
    // std::printf("size = %li\n", size);
    if (size != 41){
        std::fprintf(stderr, "ERR Synthesizer::loadConfig: FILE %s IS NOT RIGHT SIZE\n", path.c_str());
        return 2;
    }

    float fade, release, sustain, attack;
    generator_type tempGenerator;

    file.seekg(std::ios_base::beg);
    file.read(reinterpret_cast<char*>(&settings.pitch), sizeof(settings.pitch));
    file.read(reinterpret_cast<char*>(&settings.volume), sizeof(settings.volume));
    file.read(reinterpret_cast<char*>(&attack), sizeof(attack));
    file.read(reinterpret_cast<char*>(&sustain), sizeof(sustain));
    file.read(reinterpret_cast<char*>(&fade), sizeof(fade));
    file.read(reinterpret_cast<char*>(&settings.fadeTo), sizeof(settings.fadeTo));
    file.read(reinterpret_cast<char*>(&settings.rawFadeTo), sizeof(settings.rawFadeTo));
    file.read(reinterpret_cast<char*>(&release), sizeof(release));
    file.read(reinterpret_cast<char*>(&settings.dynamicsDuration), sizeof(settings.dynamicsDuration));
    file.read(reinterpret_cast<char*>(&settings.stereoMix), sizeof(settings.stereoMix));
    file.read(reinterpret_cast<char*>(&tempGenerator), sizeof(tempGenerator));


    settings.attack.set(attack, settings.sampleRate);
    settings.sustain.set(sustain, settings.sampleRate);
    settings.release.set(release, settings.sampleRate);
    settings.fade.set(fade, settings.sampleRate);

    dynamicsController.calculateDynamicsProfile(settings);
    dynamicsController.calculateReleaseProfile(settings);
    calculateStereoFactor();

    if (tempGenerator > generator_type::LAST){
        generatorType = generator_type::SINE;
        std::fprintf(stderr, "WARING Synthesizer::loadConfig: DATA MAY BE DAMAGED\n");
    }
    setGenerator(tempGenerator);

    file.close();
    return 0;
}
