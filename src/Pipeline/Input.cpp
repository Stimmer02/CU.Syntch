#include "Input.h"

using namespace pipeline;

Input::Input(){
    keyCount = 0; //indicates if object was inicialized
    running = false;
}

Input::~Input(){
    if (running){
        stopAllInputs();
    }
}

void Input::cleanup(){
    if (running){
        stopAllInputs();
    }

    synths.removeAll();
    midiInput.removeAll();
}

char Input::startAllInputs(){
    if (keyCount == 0){
        return -1;
    }

    if (running){
        return -2;
    }

    AKeyboardRecorder** allInputs = midiInput.getAll();
    for (int i = 0; i < midiInput.getElementCount(); i++){
        if (allInputs[i]->start()){
            stopAllInputs();
            return -3;
        }
//         for (uint timeout = 0; allInputs[i]->isRunning(); timeout++){
//             if (timeout >= 64){
//                 stopAllInputs();
//
//                 return -4;
//             }
//             std::this_thread::sleep_for(std::chrono::milliseconds(20));
//         }
    }

    running = true;
    return 0;
}

char Input::stopAllInputs(){
    if (keyCount == 0){
        return -1;
    }

    if (running == false){
        return -2;
    }

    char output = 0;
    AKeyboardRecorder** allInputs = midiInput.getAll();
    for (int i = 0; i < midiInput.getElementCount(); i++){
        if (allInputs[i]->stop()){
           output = -3;
        }
    }
    running = false;
    return output;
}

void Input::clearBuffers(){
    AKeyboardRecorder** allInputs = midiInput.getAll();
    for (int i = 0; i < midiInput.getElementCount(); i++){
       allInputs[i]->buffer->clearInactiveBuffer();
       allInputs[i]->buffer->swapActiveBuffer();
       allInputs[i]->buffer->clearInactiveBuffer();
    }
}

char Input::init(audioFormatInfo audioInfo, ushort keyCount){
    if (running){
        return -1;
    }

    this->audioInfo = audioInfo;
    this->keyCount = keyCount;
    if (midiInput.getElementCount() > 0 || synths.getElementCount() > 0){
        std::fprintf(stderr, "WARNING pipeline::Input::init: OBJECT CONTAINED COMPONENTS (Synthesizer_CUDA/AKeyborRecorder) BEFORE INITIALIZATION (ALL DELETED)");
        cleanup();
    }

    return 0;
}

short Input::addInput(AKeyboardRecorder* input){
    short ID = midiInput.add(input);
    if (running){
        midiInput.getElement(ID)->start();
    }
    return ID;
}

char Input::removeInput(short ID){
    if (running){
        AKeyboardRecorder* input = midiInput.getElement(ID);
        input->stop();
        while (input->isRunning());
    }
    char returnCode = midiInput.remove(ID);
    synthWithConnection** allSynths = synths.getAll();
    for (int i = 0 ; i < synths.getElementCount(); i++){
        if (allSynths[i]->midiInputID == ID){
            allSynths[i]->midiInputID = -1;
        }
    }
    return returnCode;
}

short Input::getInputCount(){
    return midiInput.getElementCount();
}

short Input::addSynthesizer(pipelineAudioBuffer_CUDA* buffer){
    synthWithConnection* newSynth = new synthWithConnection(buffer, audioInfo, keyCount);
    return synths.add(newSynth);
}

char Input::removeSynthesizer(short ID){
    return synths.remove(ID);
}

short Input::getSynthesizerCount(){
    return synths.getElementCount();
}

void Input::setSynthetiserSetting(short ID, synthesizer::settings_name settingsName, float value){
    synths.getElement(ID)->synth.setSettings(settingsName, value);
}

void Input::setSynthetiserSetting(short ID, synthesizer::generator_type type){
    synths.getElement(ID)->synth.setGenerator(type);
}


const synthesizer::settings_CUDA* Input::getSynthetiserSettings(short ID){
    return synths.getElement(ID)->synth.getSettings();
}

float Input::getSynthetiserSetting(short ID, synthesizer::settings_name settingName){
    const synthesizer::settings_CUDA& settings = *synths.getElement(ID)->synth.getSettings();
    switch (settingName) {
        case synthesizer::PITCH:
            return settings.pitch;
        case synthesizer::ATTACK:
            return settings.attack.raw;
        case synthesizer::SUSTAIN:
            return settings.sustain.raw;
        case synthesizer::FADE:
            return settings.fade.raw;
        case synthesizer::FADETO:
            return settings.fadeTo;
        case synthesizer::RELEASE:
            return settings.release.raw;
        case synthesizer::VOLUME:
            return settings.volume;
        case synthesizer::STEREO:
            return settings.stereoMix;
        case synthesizer::INVALID:
            return 0;
    }
    return 0;
}

synthesizer::generator_type Input::getSynthetiserType(const ushort& ID){
    return synths.getElement(ID)->synth.getGeneratorType();
}

char Input::connectInputToSynth(short inputID, short synthID){
    if (synths.IDValid(synthID) == false){
        return -1;
    }
    if (midiInput.IDValid(inputID) == false){
        return -2;
    }

    synths.getElement(synthID)->midiInputID = inputID;

    return 0;
}

char Input::disconnectSynth(short synthID){
    if (synths.IDValid(synthID) == false){
        return -1;
    }

    synths.getElement(synthID)->midiInputID = -2;//TODO: check if it works?
    return 0;
}

void Input::swapActiveBuffers(){
    midiInput.swapActiveBuffers();
}

void Input::cycleBuffers(){
    AKeyboardRecorder** allInputs = midiInput.getAll();
    keyboardTransferBuffer_CUDA** allBuffers = midiInput.getAllBuffers();

    for (int i = 0; i < midiInput.getElementCount(); i++){
        allInputs[i]->buffer->swapActiveBuffer();
        allBuffers[i]->convertBuffer(allInputs[i]->buffer);
        allInputs[i]->buffer->clearInactiveBuffer();
    }
}

void Input::cycleBuffers(double& swapTime, double& conversionTime){
    AKeyboardRecorder** allInputs = midiInput.getAll();
    keyboardTransferBuffer_CUDA** allBuffers = midiInput.getAllBuffers();

    std::chrono::_V2::system_clock::time_point timeStart;
    std::chrono::_V2::system_clock::time_point timeEnd;

    timeStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < midiInput.getElementCount(); i++){
        allInputs[i]->buffer->swapActiveBuffer();
    }
    timeEnd = std::chrono::high_resolution_clock::now();
    swapTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count()/1000000;

    timeStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < midiInput.getElementCount(); i++){
        allBuffers[i]->convertBuffer(allInputs[i]->buffer);
        allInputs[i]->buffer->clearInactiveBuffer();
    }
    timeEnd = std::chrono::high_resolution_clock::now();
    conversionTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count()/1000000;

}

void Input::generateSampleWith(short synthID, pipelineAudioBuffer_CUDA* buffer, keyboardTransferBuffer_CUDA* keyboardState){
    synths.getElement(synthID)->synth.generateSample(buffer, keyboardState);
}

void Input::generateSampleWith(short synthID){
    synthWithConnection& synthIter = *synths.getElement(synthID);
    if (synthIter.midiInputID >= 0){
        synthIter.synth.generateSample(synthIter.buffer, midiInput.getBuffer(synthIter.midiInputID));
    }
}

void Input::generateSamples(const std::vector<short>& synthIDs){
    for (ushort i = 0; i < synthIDs.size(); i++){
        synthWithConnection& synthIter = *synths.getElement(synthIDs.at(i));
        if (synthIter.midiInputID >= 0){
            synthIter.synth.generateSample(synthIter.buffer, midiInput.getBuffer(synthIter.midiInputID));
        }
    }
}

void Input::generateSamples(){
    for (ushort i = 0; i < synths.getElementCount(); i++){
        synthWithConnection& synthIter = *synths.getElementByIndex(i);
        if (synthIter.midiInputID >= 0){
            synthIter.synth.generateSample(synthIter.buffer, midiInput.getBuffer(synthIter.midiInputID));
        }
    }
}

char Input::saveSynthConfig(std::string path, short ID){
    return synths.getElement(ID)->synth.saveConfig(path);
}

char Input::loadSynthConfig(std::string path, short ID){
    return synths.getElement(ID)->synth.loadConfig(path);
}

bool Input::synthIDValid(short ID){
    return synths.IDValid(ID);
}

bool Input::inputIDValid(short ID){
    return midiInput.IDValid(ID);
}

void Input::reorganizeIDs(){
    midiInput.reorganizeIDs();
    synths.reorganizeIDs();
    for (short i = 0; i < synths.getElementCount(); i++){
        synths.getElement(i)->midiInputID = -2;
    }
}

void Input::removeAll(){
    midiInput.removeAll();
    synths.removeAll();
}

long Input::getActivationTimestamp(){
    if (midiInput.getElementCount() == 0){
        return 0;
    }

    return midiInput.getAll()[0]->buffer->getActivationTimestamp();
}
