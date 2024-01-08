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

char Input::init(audioFormatInfo audioInfo, ushort keyCount){
    if (running){
        return -1;
    }

    this->audioInfo = audioInfo;
    this->keyCount = keyCount;
    if (midiInput.getElementCount() > 0 || synths.getElementCount() > 0){
        std::fprintf(stderr, "WARNING pipeline::Input::init: OBJECT CONTAINED COMPONENTS (Synthesizer/AKeyborRecorder) BEFORE INITIALIZATION (ALL DELETED)");
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

short Input::addSynthesizer(){
    synthWithConnection* newSynth = new synthWithConnection(audioInfo, keyCount);
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

const synthesizer::settings* Input::getSynthetiserSettins(short ID){
    return synths.getElement(ID)->synth.getSettings();
}

char Input::connectInputToSynth(short inputID, short synthID){
    if (synths.IDValid(synthID) == false){
        return -1;
    }
    if (midiInput.IDValid(synthID) == false){
        return -2;
    }

    synths.getElement(synthID)->midiInputID = inputID;

    return 0;
}

void Input::swapActiveBuffers(){
    midiInput.swapActiveBuffers();
}

void Input::cycleBuffers(){
    static AKeyboardRecorder** allInputs = midiInput.getAll();
    static keyboardTransferBuffer** allBuffers = midiInput.getAllBuffers();

    for (int i = 0; i < midiInput.getElementCount(); i++){
        allInputs[i]->buffer->swapActiveBuffer();
        allBuffers[i]->convertBuffer(allInputs[i]->buffer);
        allInputs[i]->buffer->clearInactiveBuffer();
    }
}

void Input::generateSampleWith(short synthID, pipelineAudioBuffer* buffer, keyboardTransferBuffer* keyboardState){
    synths.getElement(synthID)->synth.generateSample(buffer, keyboardState);
}


void Input::generateSamples(pipelineAudioBuffer* temporaryBuffer){
    synths.getElement(0)->synth.generateSample(temporaryBuffer, midiInput.getBuffer(0));
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
