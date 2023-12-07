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
    convertBuffer(keyboardBuffer->getInactiveBuffer());
}

void inline keyboardTransferBuffer::convertBuffer(uchar* buff[127]){
    for (uint i = 0; i < keyCount; i++){
        for (uint j = 0; j < sampleSize; j++){
            if (buff[i][j] == 255){
                lastState[i] = 0;
            } else if (buff[i][j] > 0){
                lastState[i] = buff[i][j];
            }
            buffer[i][j] = lastState[i];
        }
    }
}


