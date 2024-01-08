#include "Input.h"

using namespace pipeline;

Input::Input(){
    keyCount = 0; //indicates if object was inicialized
}

Input::~Input(){}

void Input::cleanup(){
    synths.removeAll();
    midiInput.removeAll();
}

char Input::startAllInputs(){
    if (keyCount == 0){
        return -1;
    }
    AKeyboardRecorder** allInputs = midiInput.getAll();
    for (uint i = 0; i < midiInput.getElementCount(); i++){
        if (allInputs[i]->start()){
           return -2;
        }
    }
    return 0;
}

char Input::stopAllInputs(){
    if (keyCount == 0){
        return -1;
    }
    char output = 0;
    AKeyboardRecorder** allInputs = midiInput.getAll();
    for (uint i = 0; i < midiInput.getElementCount(); i++){
        if (allInputs[i]->stop()){
           output = -2;
        }
    }
    return output;
}

char Input::init(audioFormatInfo audioInfo, ushort keyCount){
    this->audioInfo = audioInfo;
    this->keyCount = keyCount;
    if (midiInput.getElementCount() > 0 || synths.getElementCount() > 0){
        std::fprintf(stderr, "WARNING pipeline::Input::init: OBJECT CONTAINED COMPONENTS (Synthesizer/AKeyborRecorder) BEFORE INITIALIZATION (ALL DELETED)");
        cleanup();
    }

    return 0;
}

short Input::addInput(AKeyboardRecorder* input){
    return midiInput.add(input);
}

char Input::removeInput(short ID){
    return midiInput.remove(ID);
    synthWithConnection** allSynths = synths.getAll();
    for (uint i = 0 ; i < synths.getElementCount(); i++){
        if (allSynths[i]->midiInputID == ID){
            allSynths[i]->midiInputID = -1;
        }
    }
}

short Input::getInputCount(){
    return midiInput.getElementCount();
}

void Input::swapActiveBuffers(){
    midiInput.swapActiveBuffers();
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

void Input::generateSamples(){

}

char Input::saveSynthConfig(std::string path, short id){

    return 0;
}

char Input::loadSynthConfig(std::string path, short id){
    return 0;
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
