#include "Synthesizer_CUDA.h"
#include "Synthesizer_CUDA.cu"

using namespace synthesizer;

Synthesizer_CUDA::Synthesizer_CUDA(const audioFormatInfo& audioInfo, const ushort& keyCount){
    cudaMalloc((void**)&d_settings, sizeof(settings_CUDA));

    settings.keyCount = keyCount;
    settings.sampleSize = audioInfo.sampleSize;
    settings.sampleRate = audioInfo.sampleRate;
    settings.pitch = 0;
    settings.volume = 0.15;
    dynamicsController.setDynamics(&d_settings->attack, settings.attack, 1, settings.sampleRate);
    dynamicsController.setDynamics(&d_settings->sustain, settings.sustain, 1, settings.sampleRate);
    dynamicsController.setDynamics(&d_settings->fade, settings.fade, 0.5, settings.sampleRate);
    settings.rawFadeTo = 0.8;
    settings.fadeTo = 0.8;
    settings.release.raw = 1.2;
    settings.stereoMix = 0.5;

    settings.maxValue = 0;
    uint a = 1;
    for (uint i = 1; i < audioInfo.bitDepth; i++){
        settings.maxValue += a;
        a <<= 1;
    }
    settings.maxValue -= 1;

    cudaMemcpy(d_settings, &settings, sizeof(settings_CUDA), cudaMemcpyHostToDevice);

    soundGenerator = new Generator_CUDA(settings);

    notes.init(audioInfo.sampleSize, keyCount);

    calculateFrequencies();
    calculateStereoFactor();
    dynamicsController.calculateDynamicsProfile(d_settings, settings);
    dynamicsController.calculateReleaseProfile(d_settings, settings, settings.release.raw); //really bad way to do this during main loop since it could invoke segmentation fault, however this is a constructor (the only save place)
}

Synthesizer_CUDA::~Synthesizer_CUDA(){
    cudaFree(d_settings);
    delete soundGenerator;
}

const struct settings_CUDA* Synthesizer_CUDA::getSettings(){
    return &this->settings;
}

void Synthesizer_CUDA::setSettings(const settings_name& settingsName, const float& value){
    switch (settingsName) {
        case synthesizer::PITCH:
            settings.pitch = value;
            cudaMemcpy(&(d_settings->pitch), &settings.pitch, sizeof(uint), cudaMemcpyHostToDevice);
            calculateFrequencies();
            break;

        case synthesizer::ATTACK:
            dynamicsController.setDynamics(&d_settings->attack, settings.attack, 1, settings.sampleRate);
            dynamicsController.calculateDynamicsProfile(d_settings, settings);
            break;

        case synthesizer::SUSTAIN:
            dynamicsController.setDynamics(&d_settings->sustain, settings.sustain, 1, settings.sampleRate);
            dynamicsController.calculateDynamicsProfile(d_settings, settings);
            break;

        case synthesizer::FADE:
            dynamicsController.setDynamics(&d_settings->fade, settings.fade, 0.5, settings.sampleRate);
            dynamicsController.calculateDynamicsProfile(d_settings, settings);
            break;

        case synthesizer::RELEASE:
            dynamicsController.calculateReleaseProfile(d_settings, settings, value);
            break;

        case synthesizer::VOLUME:
            settings.volume = value;
            cudaMemcpy(&(d_settings->volume), &settings.volume, sizeof(float), cudaMemcpyHostToDevice);
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
            cudaMemcpy(&(d_settings->rawFadeTo), &settings.rawFadeTo, sizeof(float), cudaMemcpyHostToDevice);
            dynamicsController.calculateDynamicsProfile(d_settings, settings);
            break;
        case synthesizer::INVALID:
            return;
    }
}

generator_type Synthesizer_CUDA::getGeneratorType(){
    return soundGenerator->getGeneratorType();
}

void Synthesizer_CUDA::setGenerator(const generator_type& type){
    if (type == soundGenerator->getGeneratorType()){
        return;
    }

    soundGenerator->setGenerator(type);
    calculateFrequencies();
}


void Synthesizer_CUDA::calculateFrequencies(){
    static const uint blockSize = 128;

    switch (soundGenerator->getGeneratorType()){
        case SINE:
            kernel_calculateFrequenciesSine<<<1, blockSize>>>(notes.getDeviceBuffer(), d_settings);
            break;
        case TRIANGLE:
            kernel_calculateFrequenciesTriangle<<<1, blockSize>>>(notes.getDeviceBuffer(), d_settings);
            break;
        case SQUARE:
        case SAWTOOTH:
        case NOISE1:
            kernel_calculateFrequenciesRest<<<1, blockSize>>>(notes.getDeviceBuffer(), d_settings);
            break;
        case INVALID_GEN:
            break;
        }
}

void Synthesizer_CUDA::calculateStereoFactor(){
    static const uint blockSize = 128;
    kernel_calculateStereoFactor<<<1, blockSize>>>(notes.getDeviceBuffer(), d_settings);
}

void Synthesizer_CUDA::mixAudio(pipelineAudioBuffer_CUDA*& audioBuffer){
    static const uint blockSize = 128;
    kernel_mixAudio<<<(settings.sampleSize + blockSize - 1) / blockSize, blockSize>>>(notes.getDeviceNoteBuffer(), d_settings, audioBuffer->d_bufferL, audioBuffer->d_bufferR);
}

void Synthesizer_CUDA::generateSample(pipelineAudioBuffer_CUDA* audioBuffer, const keyboardTransferBuffer_CUDA* keyboardState){

    soundGenerator->generate(notes.getDeviceNoteBuffer(), keyboardState->d_buffer, d_settings, settings, dynamicsController.getDynamicsProfile(), dynamicsController.getReleaseProfile(), dynamicsController.getReleaseToAttackIndexMap());

    mixAudio(audioBuffer);
}

char Synthesizer_CUDA::saveConfig(std::string path){
    std::ofstream file(path, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    if (file.fail()){
        std::fprintf(stderr, "ERR Synthesizer_CUDA::saveConfig: CANNOT OPEN FILE %s\n", path.c_str());
        return 1;
    }
    generator_type generatorType = soundGenerator->getGeneratorType();

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

char Synthesizer_CUDA::loadConfig(std::string path){
    std::ifstream file(path, std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
    if (file.fail()){
        std::fprintf(stderr, "ERR Synthesizer_CUDA::loadConfig: CANNOT OPEN FILE %s\n", path.c_str());
        return 1;
    }
    long size = file.tellg();
    // std::printf("size = %li\n", size);
    if (size != 41){
        std::fprintf(stderr, "ERR Synthesizer_CUDA::loadConfig: FILE %s IS NOT RIGHT SIZE\n", path.c_str());
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

    dynamicsController.setDynamics(&d_settings->attack, settings.attack, attack, settings.sampleRate);
    dynamicsController.setDynamics(&d_settings->sustain, settings.sustain, sustain, settings.sampleRate);
    dynamicsController.setDynamics(&d_settings->fade, settings.fade, fade, settings.sampleRate);

    cudaMemcpy(d_settings, &settings, sizeof(settings_CUDA), cudaMemcpyHostToDevice);

    dynamicsController.calculateDynamicsProfile(d_settings, settings);
    dynamicsController.calculateReleaseProfile(d_settings, settings, release);
    calculateStereoFactor();

    if (tempGenerator > generator_type::LAST){
        soundGenerator->setGenerator(generator_type::SINE);
        std::fprintf(stderr, "WARING Synthesizer_CUDA::loadConfig: DATA MAY BE DAMAGED\n");
    }
    setGenerator(tempGenerator);

    file.close();
    return 0;
}



