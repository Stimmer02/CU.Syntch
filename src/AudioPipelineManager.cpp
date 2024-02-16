#include "AudioPipelineManager.h"

using namespace pipeline;

AudioPipelineManager::AudioPipelineManager(audioFormatInfo audioInfo, ushort keyCount): audioInfo(audioInfo), keyCount(keyCount), component(&this->audioInfo){
    running = false;
    pipelineThread = nullptr;

    statisticsService = new statistics::PipelineStatisticsService(audioInfo.sampleSize*long(1000000)/audioInfo.sampleRate, 64, audioInfo, 0);

    input.init(audioInfo, keyCount);
    output.init(audioInfo);

    outputBuffer = nullptr;
}

AudioPipelineManager::~AudioPipelineManager(){
    stop();
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
    if (outputBuffer == nullptr){
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
    executionQueue.build(componentQueues, outputBuffer);
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
    static const std::vector<audioBufferQueue*>& backwardsExecution = executionQueue.getQueue();

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
        for (int i = backwardsExecution.size() - 1; i >= 0; i--){
            component.applyEffects(backwardsExecution[i]);
        }

        statisticsService->loopWorkEnd();

        output.play(&outputBuffer->buffer);
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
            return component.components.IDValid(ID);
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
    audioBufferQueue* newQueue = new audioBufferQueue(pipeline::SYNTH, audioInfo.sampleSize);
    short newSynthID = input.addSynthesizer(&newQueue->buffer);
    newQueue->parentID = newSynthID;
    componentQueues.push_back(newQueue);
    return newSynthID;
}

char AudioPipelineManager::removeSynthesizer(short ID){
    for (uint i = 0; i < componentQueues.size(); i++){
        if (componentQueues.at(i)->parentID == ID && componentQueues.at(i)->parentType == pipeline::SYNTH){
            if (outputBuffer != nullptr && outputBuffer->parentID == ID && outputBuffer->parentType == pipeline::SYNTH){
                std::printf("WARNING: REMOVING OUTPUT BUFFER\n");
                if (running){
                    stop();
                    std::printf("PIPELINE STOPPED\n");
                }
                outputBuffer = nullptr;
            }
            for (uint j = 0; j < componentQueues.at(i)->componentIDQueue.size(); j++){
                component.components.getElement(componentQueues.at(i)->componentIDQueue.at(j))->includedIn = nullptr;
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
        audioBufferQueue& queueIterator = *componentQueues.at(i);
        if (queueIterator.parentID == ID && queueIterator.parentType == IDType){
            outputBuffer = &queueIterator;
            return 0;
        }
    }

    return -3;
}

short AudioPipelineManager::addComponent(component_type type){
    return component.addComponent(type);
}

char AudioPipelineManager::removeComponent(short ID){
    disconnectCommponent(ID);
    return component.components.remove(ID);
}

short AudioPipelineManager::getComponentCout(){
    return component.components.getElementCount();
}

char AudioPipelineManager::connectComponent(short componentID, ID_type parentType, short parentID){
    if (component.components.IDValid(componentID) == false){
        return -1;
    }
    if (IDValid(parentType, parentID) == false){
        return -2;
    }

    for (uint i = 0; i < componentQueues.size(); i++){ //REALLY INEFFICENT, but bearable
        audioBufferQueue& queue = *componentQueues.at(i);
        if (queue.parentID == parentID && queue.parentType == parentType){
            AComponent& tempComponent = *component.components.getElement(componentID);
            disconnectCommponent(componentID);
            queue.componentIDQueue.push_back(componentID);
            tempComponent.includedIn = &queue;
            break;
        }
    }

    return 0;
}

char AudioPipelineManager::disconnectCommponent(short componentID){
    if (component.components.IDValid(componentID) == false){
        return -1;
    }

    AComponent& tempComponent = *component.components.getElement(componentID);
    if (tempComponent.includedIn != nullptr){
        audioBufferQueue& oldQueue = *tempComponent.includedIn;
        for (uint i = 0; i < oldQueue.componentIDQueue.size(); i++){//TODO: (i < oldQueue.componentIDQueue.size()) changed to if statement below will detect system bugs
            if (oldQueue.componentIDQueue.at(i) == componentID){
                oldQueue.componentIDQueue.erase(oldQueue.componentIDQueue.begin() + i);
                break;
            }
        }
    }

    return 0;
}

char AudioPipelineManager::getComponentConnection(short componentID, ID_type& parentType, short& parentID){
    if (component.components.IDValid(componentID) == false){
        return -1;
    }

    AComponent& tempComponent = *component.components.getElement(componentID);

    if (tempComponent.includedIn == nullptr){
        parentType = pipeline::INVALID;
        parentID = -2;
    } else {
        parentType = tempComponent.includedIn->parentType;
        parentID = tempComponent.includedIn->parentID;
    }

    return 0;
}

char AudioPipelineManager::setComponentSetting(short componentID, uint settingIndex, float value){
    if (component.components.IDValid(componentID) == false){
        return -1;
    }
    AComponent& tempComponent = *component.components.getElement(componentID);

    if (tempComponent.getSettings()->count <= settingIndex){
        return -2;
    }

    tempComponent.set(settingIndex, value);

    return 0;
}

const componentSettings* AudioPipelineManager::getComopnentSettings(short componentID){
    if (component.components.IDValid(componentID) == false){
        return nullptr;
    }

    return component.components.getElement(componentID)->getSettings();
}
