#include "KeyboardDoubleBuffer.h"
#include <cstring>

KeyboardDoubleBuffer::KeyboardDoubleBuffer(const uint& sampleSize, const ushort& keyCount) : keyCount(keyCount), sampleSize(sampleSize){
    buffer[0] = new uchar*[keyCount];
    buffer[1] = new uchar*[keyCount];
    for (uint i = 0; i < keyCount; i++){
        buffer[0][i] = new uchar[sampleSize];
        buffer[1][i] = new uchar[sampleSize];
    }
    activeBuffer = 0;
    swapActiveBuffer();
    clearInactiveBuffer();
    swapActiveBuffer();
    clearInactiveBuffer();
}

KeyboardDoubleBuffer::~KeyboardDoubleBuffer(){
    for (uint i = 0; i < keyCount; i++){
        delete[] buffer[0][i];
        delete[] buffer[1][i];
    }
    delete[] buffer[0];
    delete[] buffer[1];
}

uchar** KeyboardDoubleBuffer::getInactiveBuffer(){
    return buffer[!activeBuffer];
}

uchar** KeyboardDoubleBuffer::getActiveBuffer(){
    return buffer[activeBuffer];
}

long KeyboardDoubleBuffer::getActivationTimestamp(){
    return activationTimestamp[activeBuffer];
}


void KeyboardDoubleBuffer::swapActiveBuffer(){
    activationTimestamp[!activeBuffer] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
    activeBuffer = !activeBuffer;
}

void KeyboardDoubleBuffer::clearInactiveBuffer(){
    for (uint i = 0; i < keyCount; i++){
        std::memset(buffer[!activeBuffer][i], 0, sampleSize);
    }
}

ushort KeyboardDoubleBuffer::getKeyCount(){
    return keyCount;
}

uint KeyboardDoubleBuffer::getSampleSize(){
    return sampleSize;
}
