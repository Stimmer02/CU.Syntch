#include "audioBuffer.h"

audioBuffer::audioBuffer(const uint32_t& buffSize):size(buffSize){
    buff = new uint8_t[buffSize];
    count = 0;
}

audioBuffer::~audioBuffer(){
    delete[] buff;
}
