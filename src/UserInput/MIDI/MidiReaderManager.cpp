#include "MidiReaderManager.h"

using namespace MIDI;

MidiReaderManager::MidiReaderManager(const audioFormatInfo* audioInfo) : audioInfo(audioInfo){
    playCounter = 0;
}

void MidiReaderManager::rewind(){
    for (uint i = 0; i < midiReaders.size(); i++){
        if (midiReaders[i]->isReady()){
           midiReaders[i]->revind();
        }
    }
}

char MidiReaderManager::rewind(short inputID){
    if (midiReadersMap.find(inputID) == midiReadersMap.end()){
        return 1;
    }
    return midiReadersMap[inputID]->revind();;
}

bool MidiReaderManager::isMidiFileReader(short inputID){
    if (midiReadersMap.find(inputID) == midiReadersMap.end()){
        return false;
    }
    return true;
}

char MidiReaderManager::setFile(short inputID, std::string filePath){
    if (midiReadersMap.find(inputID) == midiReadersMap.end()){
        return 1;
    }
    return midiReadersMap[inputID]->reInitFile(filePath);;
}

void MidiReaderManager::add(KeyboardRecorder_MidiFile* midiReader, short inputID){
    midiReadersMap[inputID] = midiReader;
    midiReaders.push_back(midiReader);
    midiReader->setPlayCounter(&playCounter);
}

void MidiReaderManager::remove(short inputID){
    midiReadersMap.erase(inputID);
    for (uint i = 0; i < midiReaders.size(); i++){
        if (midiReaders[i] == midiReadersMap[inputID]){
            midiReaders.erase(midiReaders.begin() + i);
            break;
        }
    }   
}

void MidiReaderManager::play(){
    for (uint i = 0; i < midiReaders.size(); i++){
        if (midiReaders[i]->isReady()){
           midiReaders[i]->play();
        }
    }
}

char MidiReaderManager::play(short inputID){
    if (midiReadersMap.find(inputID) == midiReadersMap.end()){
        return 1;
    }
    return midiReadersMap[inputID]->play();
}

void MidiReaderManager::pause(){
    for (uint i = 0; i < midiReaders.size(); i++){
        if (midiReaders[i]->isReady()){
           midiReaders[i]->pause();
        }
    }
}

char MidiReaderManager::pause(short inputID){
    if (midiReadersMap.find(inputID) == midiReadersMap.end()){
        return 1;
    }
    return midiReadersMap[inputID]->pause();
}

std::string MidiReaderManager::getFile(short inputID){
    if (midiReadersMap.find(inputID) == midiReadersMap.end()){
        return "";
    }
    return midiReadersMap[inputID]->getPath();
}

void MidiReaderManager::printReaders(){
    for (const auto& readerPair : midiReadersMap) {
        std::printf("MIDI(%d): %s\n", readerPair.first, readerPair.second->getPath().c_str());
    }
}

short MidiReaderManager::getCount(){
    return midiReaders.size();
}

int MidiReaderManager::getPlayCounter(){
    return playCounter;
}