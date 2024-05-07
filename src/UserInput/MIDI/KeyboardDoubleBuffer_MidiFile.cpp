#include "KeyboardDoubleBuffer_MidiFile.h"

using namespace MIDI;

KeyboardDoubleBuffer_MidiFile::KeyboardDoubleBuffer_MidiFile(const std::string path, const uint& sampleSize, const uint& sampleRate): fileReader(path, sampleSize, sampleRate), sampleSize(sampleSize){
    swapActiveBuffer();
}

KeyboardDoubleBuffer_MidiFile::~KeyboardDoubleBuffer_MidiFile(){
    fileReader.close();
}

uchar** KeyboardDoubleBuffer_MidiFile::getInactiveBuffer(){
    return fileReader.getTempBuffer();
}

uchar** KeyboardDoubleBuffer_MidiFile::getActiveBuffer(){
    return fileReader.getTempBuffer();
}

long KeyboardDoubleBuffer_MidiFile::getActivationTimestamp(){
    return activationTimestamp;
}

void KeyboardDoubleBuffer_MidiFile::swapActiveBuffer(){
    activationTimestamp = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
    fileReader.fillBuffer(0);
    fileReader.notifyObserver();
}

void KeyboardDoubleBuffer_MidiFile::clearInactiveBuffer(){}

ushort KeyboardDoubleBuffer_MidiFile::getKeyCount(){
    return keyCount;
}

uint KeyboardDoubleBuffer_MidiFile::getSampleSize(){
    return sampleSize;
}
