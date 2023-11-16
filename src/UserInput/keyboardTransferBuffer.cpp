#include "keyboardTransferBuffer.h"
// #include <cstdio>

keyboardTransferBuffer::keyboardTransferBuffer(const uint& sampleSize, const ushort& keyCount) : sampleSize(sampleSize), keyCount(keyCount){
    buffer = new uchar*[sampleSize];
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
    for (uint i = 0; i < sampleSize; i++){
        for (uint j = 0; j < keyCount; j++){
            if (kBuffer[i][j] == 1){
                lastState[j] = 0;
                // std::printf("k:%d; s:%d; v:%d\n", j,i,kBuffer[i][j]);
            } else if (kBuffer[i][j] > 1){
                lastState[j] = 1;
                // std::printf("k:%d; s:%d; v:%d\n", j,i,kBuffer[i][j]);
            }
            buffer[j][i] = lastState[j];
        }
    }
}

