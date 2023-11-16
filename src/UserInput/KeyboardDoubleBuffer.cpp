#include "KeyboardDoubleBuffer.h"

KeyboardDoubleBuffer::KeyboardDoubleBuffer(const uint& sampleSize, const ushort& keyCount) : sampleSize(sampleSize), keyCount(keyCount){
    buffer[0] = new uchar*[sampleSize];
    buffer[1] = new uchar*[sampleSize];
    for (uint i = 0; i < sampleSize; i++){
        buffer[0][i] = new uchar[keyCount];
        buffer[1][i] = new uchar[keyCount];
    }
    activeBuffer = 0;
    clearInactiveBuffer();
    activeBuffer = 1;
    clearInactiveBuffer();
}

KeyboardDoubleBuffer::~KeyboardDoubleBuffer(){
    for (uint i = 0; i < sampleSize; i++){
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
    for (uint i = 0; i < sampleSize; i++){
        for (uint j = 0; j < keyCount; j++){
            buffer[!activeBuffer][i][j] = 0;
        }
    }
}
