#include "audioBuffer.h"

audioBuffer::audioBuffer(const uint16_t& buffSize):size(buffSize){
    buff = new uint8_t[buffSize];
}

audioBuffer::~audioBuffer(){
    delete[] buff;
}