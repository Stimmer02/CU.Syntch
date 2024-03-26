#include "audioFormatInfo.h"
#include <string>

unsigned long audioFormatInfo::length(){
    return fileLength/byteRate;
}

unsigned long audioFormatInfo::timeElapsed(uint32_t bytesRead){
    return bytesRead/byteRate;
}

std::string audioFormatInfo::secondsToString(unsigned long seconds){
    unsigned int minutes = seconds/60;
    seconds = seconds%60;
    std::string out = "";
    if (minutes < 10){
        out += "0";
    }
    out += std::to_string(minutes);
    out += ':';
    if (seconds < 10){
        out += "0";
    }
    if (seconds == 0){
        out += "0";
    } else {
        out += std::to_string(seconds);
    }
    return out;
}

void audioFormatInfo::print(){
    printf("type: %s\n", type.c_str());
    printf("channels: %u\nbitDepth: %d\nblockAlign: %d\nsampleRate: %u\nbyteRate: %u\nfileLength: %u\n", channels, bitDepth, blockAlign, sampleRate, byteRate, fileLength);
    if (littleEndian){
        printf("littleEndian\n");
    } else {
        printf("bigEndian\n");
    }
}
