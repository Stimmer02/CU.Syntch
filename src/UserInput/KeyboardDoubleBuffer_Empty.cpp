#include "KeyboardDoubleBuffer_Empty.h"
#include <cstring>

KeyboardDoubleBuffer_Empty::KeyboardDoubleBuffer_Empty(const uint& sampleSize, const ushort& keyCount) : keyCount(keyCount), sampleSize(sampleSize){
    buffer = new uchar*[keyCount];
    for (uint i = 0; i < keyCount; i++){
        buffer[i] = new uchar[sampleSize];
    }
    swapActiveBuffer();
    clearInactiveBuffer();
}

KeyboardDoubleBuffer_Empty::~KeyboardDoubleBuffer_Empty(){
    for (uint i = 0; i < keyCount; i++){
        delete[] buffer[i];
    }
    delete[] buffer;
}

uchar** KeyboardDoubleBuffer_Empty::getInactiveBuffer(){
    return buffer;
}

uchar** KeyboardDoubleBuffer_Empty::getActiveBuffer(){
    return buffer;
}

long KeyboardDoubleBuffer_Empty::getActivationTimestamp(){
    return activationTimestamp;
}


void KeyboardDoubleBuffer_Empty::swapActiveBuffer(){
    activationTimestamp = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
}

void KeyboardDoubleBuffer_Empty::clearInactiveBuffer(){
    for (uint i = 0; i < keyCount; i++){
        std::memset(buffer[i], 0, sampleSize);
    }
}

ushort KeyboardDoubleBuffer_Empty::getKeyCount(){
    return keyCount;
}

uint KeyboardDoubleBuffer_Empty::getSampleSize(){
    return sampleSize;
}
