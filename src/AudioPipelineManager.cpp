#include "AudioPipelineManager.h"
#include "Pipeline/pipelineAudioBuffer.h"

using namespace pipeline;

AudioPipelineManager::AudioPipelineManager(audioFormatInfo audioInfo, ushort keyCount): audioInfo(audioInfo), keyCount(keyCount){
    running = false;
    pipelineThread = nullptr;
    statisticsService = new statistics::PipelineStatisticsService(audioInfo.sampleSize*long(1000000)/audioInfo.sampleRate, 64, audioInfo, 0);

    input.init(audioInfo, keyCount);
    output.init(audioInfo);

    temporaryBuffer = new pipelineAudioBuffer(audioInfo.sampleSize);
}

AudioPipelineManager::~AudioPipelineManager(){
    delete statisticsService;
    delete temporaryBuffer;
}

char AudioPipelineManager::start(){
    if (running) return -1;
    if (input.startAllInputs()){
        input.stopAllInputs();
        return -2;
    }

    if (pipelineThread != nullptr){
        delete pipelineThread;
    }
    pipelineThread = new std::thread(&AudioPipelineManager::pipelineThreadFunction, this);

    running = true;
    return 0;
}

void AudioPipelineManager::stop(){

}

const statistics::pipelineStatistics* AudioPipelineManager::getStatistics(){
    return statisticsService->getStatistics();
}

const audioFormatInfo* AudioPipelineManager::getAudioInfo(){
    return &audioInfo;
}

void AudioPipelineManager::pipelineThreadFunction(){
    ulong sampleTimeLength = audioInfo.sampleSize*long(1000000)/audioInfo.sampleRate;

    // midiInput->buffer->swapActiveBuffer();
    input.swapActiveBuffers();

    // ulong nextLoop = midiInput->buffer->getActivationTimestamp() + sampleTimeLength;
    ulong nextLoop = input.getActivationTimestamp() + sampleTimeLength;

    statisticsService->firstInvocation();

    while (running){
        std::this_thread::sleep_until(std::chrono::time_point<std::chrono::system_clock>(std::chrono::nanoseconds((nextLoop)*1000)));
        statisticsService->loopStart();

        // midiInput->buffer->swapActiveBuffer();
        input.swapActiveBuffers();

        nextLoop += sampleTimeLength;

        //keyboardState->convertBuffer(midiInput->buffer);
        //midiInput->buffer->clearInactiveBuffer();
        //synth->generateSample(pipelineBuffer, keyboardState);

        // printLastBuffer(pipelineBuffer->bufferL, pipelineBuffer->size);

        statisticsService->loopWorkEnd();
        // bufferConverter->toPCM(pipelineBuffer, buffer);
        // audioOutput->playBuffer(buffer);
        // if (recording){
        //     audioRecorder.saveBuffer(buffer);
        // }
        output.play(temporaryBuffer);
    }
}

void AudioPipelineManager::recordUntilStreamEmpty(MIDI::MidiFileReader& midi, std::string filename){

}

