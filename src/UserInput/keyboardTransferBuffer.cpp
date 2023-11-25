#include "keyboardTransferBuffer.h"

keyboardTransferBuffer::keyboardTransferBuffer(const uint& sampleSize, const ushort& keyCount) : sampleSize(sampleSize), keyCount(keyCount){
    buffer = new uchar*[keyCount];
    for (uint i = 0; i < keyCount; i++){
        buffer[i] = new uchar[sampleSize];
    }

    lastState = new uchar[keyCount];
    for (uint i = 0; i < keyCount; i++){
        lastState[i] = 0;
    }
}

keyboardTransferBuffer::~keyboardTransferBuffer(){
    for (uint i = 0; i < keyCount; i++){
        delete[] buffer[i];
    }
    delete[] buffer;
    delete[] lastState;
}

void keyboardTransferBuffer::convertBuffer(KeyboardDoubleBuffer* keyboardBuffer){
    uchar** kBuffer = keyboardBuffer->getInactiveBuffer();
    for (uint i = 0; i < keyCount; i++){
        for (uint j = 0; j < sampleSize; j++){
            if (kBuffer[i][j] == 1){
                lastState[i] = 0;
            } else if (kBuffer[i][j] > 1){
                lastState[i] = 1;
            }
            buffer[i][j] = lastState[i];//TODO:ELIMINATE THIS BUG
        }
    }
}

