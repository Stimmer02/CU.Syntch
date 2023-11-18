#include "AudioPipelineSubstitute.h"
#include "Pipeline/Statistics/PipelineStatisticsService.h"
#include "Synthesizer/settings.h"

AudioPipelineSubstitute::AudioPipelineSubstitute(audioFormatInfo audioInfo, ushort keyCount, AKeyboardRecorder* midiInput){
    this->audioInfo = audioInfo;
    this-> keyCount = keyCount;
    this->midiInput = midiInput;

    keyboardState = new keyboardTransferBuffer(audioInfo.sampleSize, keyCount);
    audioOutput = new OutStream_PulseAudio();
    synth = new synthesizer::Synthesizer(audioInfo, keyCount);

    pipelineBuffer = new pipelineAudioBuffer(audioInfo.sampleSize);
    buffer = new audioBuffer(audioInfo.sampleSize*audioInfo.bitDepth/8);
    buffer->count = buffer->size;

    running = false;
    recording = false;
    pipelineThread = nullptr;

    if (audioInfo.channels == 1){
        if (audioInfo.bitDepth <= 8){
            bufferConverter = new BufferConverter_Mono8();
        } else if (audioInfo.bitDepth <= 16){
            bufferConverter = new BufferConverter_Mono16();
        } else if (audioInfo.bitDepth <= 24){
            bufferConverter = new BufferConverter_Mono24();
        } else if (audioInfo.bitDepth <= 32){
            bufferConverter = new BufferConverter_Mono32();
        } else {
            bufferConverter = nullptr;
            std::fprintf(stderr, "ERR AudioPipelineSubstitute::AudioPipelineSubstitute: UNSUPPORTED BIT DEPTH\n");
            exit(1);
        }
    } else {
        bufferConverter = nullptr;
        std::fprintf(stderr, "ERR AudioPipelineSubstitute::AudioPipelineSubstitute: UNSUPPORTED CHANNEL COUNT\n");
        exit(1);
    }
    statisticsService = new statistics::PipelineStatisticsService(audioInfo.sampleSize*long(1000000)/audioInfo.sampleRate, 64, audioInfo, 0);
}

AudioPipelineSubstitute::~AudioPipelineSubstitute(){
    stop();

    if (bufferConverter != nullptr){
        delete bufferConverter;
    }

    delete statisticsService;
    delete keyboardState;
    delete audioOutput;
    delete synth;
    delete pipelineBuffer;
    delete buffer;
}

void AudioPipelineSubstitute::start(){
    if (running) return;
    if (audioOutput->init(audioInfo, "Synth", "Synthesizer")) return;
    if (midiInput->start()) return;
    while (midiInput->isRunning() == false);

    if (pipelineThread != nullptr){
        delete pipelineThread;
    }
    pipelineThread = new std::thread(&AudioPipelineSubstitute::pipelineThreadFunction, this);

    running = true;
}

void AudioPipelineSubstitute::stop(){
    if (running == false){
        return;
    }
    running = false;
    if (pipelineThread->joinable()){
        pipelineThread->join();
    }
    midiInput->stop();
}

void AudioPipelineSubstitute::startRecording(){
    if (recording) {
        return;
    }
    recording = true;
}

void AudioPipelineSubstitute::stopRecording(){
    if (recording == false) {
        return;
    }
    recording = false;
}

void AudioPipelineSubstitute::pipelineThreadFunction(){
    ulong sampleTimeLength = audioInfo.sampleSize*long(1000000)/audioInfo.sampleRate;
    midiInput->buffer->swapActiveBuffer();
    ulong nextLoop = midiInput->buffer->getActivationTimestamp() + sampleTimeLength;
    statisticsService->firstInvocation();

    while (running){
        std::this_thread::sleep_until(std::chrono::time_point<std::chrono::system_clock>(std::chrono::nanoseconds((nextLoop)*1000)));
        statisticsService->loopStart();
        midiInput->buffer->swapActiveBuffer();


        nextLoop += sampleTimeLength;
        keyboardState->convertBuffer(midiInput->buffer);
        midiInput->buffer->clearInactiveBuffer();


        synth->generateSample(pipelineBuffer, keyboardState);
        bufferConverter->toPCM(pipelineBuffer, buffer);
        statisticsService->loopWorkEnd();
        audioOutput->playBuffer(buffer);
    }
}

const statistics::pipelineStatistics* AudioPipelineSubstitute::getStatistics(){
    return statisticsService->getStatistics();
}

const audioFormatInfo* AudioPipelineSubstitute::getAudioInfo(){
    return &audioInfo;
}

const synthesizer::settings* AudioPipelineSubstitute::getSynthSettings(ushort id){
    return synth->getSettings();
}

// template<typename T>
void AudioPipelineSubstitute::setSynthSettings(ushort id, synthesizer::settings_name settingsName, double value){
    static synthesizer::settings* settings = synth->getSettings();
    switch (settingsName) {
        case synthesizer::PITCH:
            synth->setPitch(value);
            settings->pitch = value;
            break;

        case synthesizer::ATTACK:
            settings->attack.set(value, this->audioInfo.sampleRate);
            break;

        case synthesizer::SUSTAIN:
            settings->sustain.set(value, this->audioInfo.sampleRate);
            break;

        case synthesizer::FADE:
            settings->fade.set(value, this->audioInfo.sampleRate);
            break;

        case synthesizer::RELEASE:
            settings->release.set(value, this->audioInfo.sampleRate);
            break;

        case synthesizer::VOLUME:
            settings->volume = value;
            break;
    }
}


