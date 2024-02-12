#include "AudioPipelineManager.h"


using namespace pipeline;

AudioPipelineManager::AudioPipelineManager(audioFormatInfo audioInfo, ushort keyCount): audioInfo(audioInfo), keyCount(keyCount),  component(&audioInfo){
    running = false;
    pipelineThread = nullptr;
    statisticsService = new statistics::PipelineStatisticsService(audioInfo.sampleSize*long(1000000)/audioInfo.sampleRate, 64, audioInfo, 0);

    input.init(audioInfo, keyCount);
    output.init(audioInfo);

    outputQueue = nullptr;
}

AudioPipelineManager::~AudioPipelineManager(){
    delete statisticsService;
    for (uint i = 0; i < componentQueues.size(); i++){
        delete componentQueues.at(i);
    }
}

char AudioPipelineManager::start(){
    if (running){
        std::fprintf(stderr, "ERR: AudioPipelineManager::start PIPELINE ALREADY RUNNING\n");
        return -1;
    }
    if (outputQueue == nullptr){
        std::fprintf(stderr, "ERR: AudioPipelineManager::start OUTPUT BUFFER IS NOT SET\n");
        return -2;
    }
    if (input.startAllInputs()){
        input.stopAllInputs();
        std::fprintf(stderr, "ERR: AudioPipelineManager::start COULD NOT START ALL INPUTS\n");
        return -3;
    }

    if (pipelineThread != nullptr){
        delete pipelineThread;
    }
    executionQueue.build(componentQueues, outputQueue);
    pipelineThread = new std::thread(&AudioPipelineManager::pipelineThreadFunction, this);

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
    running = true;
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

        input.cycleBuffers();

        input.generateSamples(executionQueue.getConnectedSynthIDs());
        executeQueue();

        statisticsService->loopWorkEnd();

        output.play(&outputQueue->buffer);
    }
}

void AudioPipelineManager::executeQueue(){
    static const std::vector<AudioBufferQueue*>& backwardsExecution = executionQueue.getQueue();
    for (int i = backwardsExecution.size() - 1; i >= 0; i--){
        component.applyEffects(backwardsExecution[i]);
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
        default:
            return false;
    }
}

void AudioPipelineManager::reorganizeIDs(){
    input.reorganizeIDs();
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
    AudioBufferQueue* newQueue = new AudioBufferQueue(pipeline::SYNTH, audioInfo.sampleSize);
    short newSynthID = input.addSynthesizer(&newQueue->buffer);
    newQueue->parentID = newSynthID;
    componentQueues.push_back(newQueue);
    return newSynthID;
}

char AudioPipelineManager::removeSynthesizer(short ID){
    for (uint i = 0; i < componentQueues.size(); i++){
        if (componentQueues.at(i)->parentID == ID){
            if (outputQueue != nullptr && outputQueue->parentID == ID && outputQueue->parentType == pipeline::SYNTH){
                std::printf("WARNING: REMOVING OUTPUT BUFFER\n");
                if (running){
                    stop();
                    std::printf("WARNING: SYSTEM STOPPED\n");
                }
                outputQueue = nullptr;
            }
            delete componentQueues.at(i);
            componentQueues.erase(componentQueues.begin() + i);
            break;
        }
    }
    return input.removeSynthesizer(ID);
}

short AudioPipelineManager::getSynthesizerCount(){
    return input.getSynthesizerCount();
}

char AudioPipelineManager::connectInputToSynth(short inputID, short synthID){
    return input.connectInputToSynth(inputID, synthID);
}

char AudioPipelineManager::disconnectSynth(short synthID){
    return input.disconnectSynth(synthID);
}

const synthesizer::settings* AudioPipelineManager::getSynthSettings(const ushort& ID){
    return input.getSynthetiserSettings(ID);
}

float AudioPipelineManager::getSynthSetting(const ushort& ID, synthesizer::settings_name settingName){
    return input.getSynthetiserSetting(ID, settingName);
}

synthesizer::generator_type AudioPipelineManager::getSynthType(const ushort& ID){
    return input.getSynthetiserType(ID);
}

void AudioPipelineManager::setSynthSetting(const ushort& ID, const synthesizer::settings_name& settingsName, const float& value){
    input.setSynthetiserSetting(ID, settingsName, value);
}

void AudioPipelineManager::setSynthSetting(const ushort& ID, const synthesizer::generator_type& type){
    input.setSynthetiserSetting(ID, type);
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


char AudioPipelineManager::pauseInput(){
    return input.stopAllInputs();
}

char AudioPipelineManager::reausumeInput(){
    return input.startAllInputs();
}

void AudioPipelineManager::clearInputBuffers(){
    input.clearBuffers();
}


//EFFECT CONTROLL

char AudioPipelineManager::setOutputBuffer(short ID, ID_type IDType){
    if (IDValid(IDType, ID) == false){
        return -1;
    }

    if (IDType != SYNTH && IDType != COMP){
       return -2;
    }

    for (uint i = 0; i < componentQueues.size(); i++){
        AudioBufferQueue& queueIterator = *componentQueues.at(i);
        if (queueIterator.parentID == ID && queueIterator.parentType == IDType){
            outputQueue = &queueIterator;
            return 0;
        }
    }

    return -3;
}
