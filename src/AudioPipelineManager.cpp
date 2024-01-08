#include "AudioPipelineManager.h"
#include "Pipeline/pipelineAudioBuffer.h"
#include "Synthesizer.h"
#include "UserInput/keyboardTransferBuffer.h"

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
    if (running == false){
        return;
    }
    running = false;
    output.stopRecording();
    if (pipelineThread->joinable()){
        pipelineThread->join();
    }
    input.stopAllInputs();
}

bool AudioPipelineManager::isRuning(){
    return running;
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
        nextLoop += sampleTimeLength;

        //midiInput->buffer->swapActiveBuffer();
        //keyboardState->convertBuffer(midiInput->buffer);
        //midiInput->buffer->clearInactiveBuffer();
        input.cycleBuffers();

        //synth->generateSample(pipelineBuffer, keyboardState);
        input.generateSamples(temporaryBuffer);

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

char AudioPipelineManager::recordUntilStreamEmpty(MIDI::MidiFileReader& midi, short synthID, std::string filename){
    if (running){
        return 1;
    }

    if (input.synthIDValid(synthID) == false){
        return 3;
    }

    keyboardTransferBuffer* keyboardState = new keyboardTransferBuffer(audioInfo.sampleSize, keyCount);
    pipelineAudioBuffer* pipelineBuffer = new pipelineAudioBuffer(audioInfo.sampleSize);

    midi.rewindFile();
    if (filename.empty()){
        output.startRecording();
    } else {
        output.startRecording(filename);
    }

    while (midi.isFileReady() && !midi.eofChunk(0)){
        midi.fillBuffer(keyboardState, 0);
        input.generateSampleWith(synthID, pipelineBuffer, keyboardState);
        output.onlyRecord(pipelineBuffer);
    }

    midi.fillBuffer(keyboardState, 0);
    for (uint i = 0; i <= 2*audioInfo.sampleRate/audioInfo.sampleSize; i++){
        input.generateSampleWith(synthID, pipelineBuffer, keyboardState);
        output.onlyRecord(pipelineBuffer);
    }
    output.stopRecording();

    delete keyboardState;
    delete pipelineBuffer;

    return 0;
}

bool AudioPipelineManager::IDValid(pipeline::ID_type type, short ID){
    switch (type) {

    case INPUT:
        return input.inputIDValid(ID);

    case SYNTH:
        return input.synthIDValid(ID);

    case COMP:
        return false;
    }
    return false;
}


//OUTPUT
char AudioPipelineManager::startRecording(){
    return output.startRecording();
}

char AudioPipelineManager::startRecording(std::string filename){
    return output.startRecording(filename);
}

char AudioPipelineManager::stopRecording(){
    return output.stopRecording();
}

bool AudioPipelineManager::isRecording(){
    return output.isRecording();
}


//INPUT
char AudioPipelineManager::saveSynthConfig(std::string path, ushort ID){
    return input.saveSynthConfig(path, ID);
}

char AudioPipelineManager::loadSynthConfig(std::string path, ushort ID){
    return input.loadSynthConfig(path, ID);
}

short AudioPipelineManager::addSynthesizer(){
    return input.addSynthesizer();
}

char AudioPipelineManager::removeSynthesizer(short ID){
    return input.removeSynthesizer(ID);
}

short AudioPipelineManager::getSynthesizerCount(){
    return input.getSynthesizerCount();
}


short AudioPipelineManager::addInput(AKeyboardRecorder*& newInput){
    return input.addInput(newInput);
}

char AudioPipelineManager::removeInput(short ID){
    return input.removeInput(ID);
}

short AudioPipelineManager::getInputCount(){
    return input.getInputCount();
}


void AudioPipelineManager::pauseInput(){
    input.stopAllInputs();
}

void AudioPipelineManager::reausumeInput(){
    input.startAllInputs();
}
