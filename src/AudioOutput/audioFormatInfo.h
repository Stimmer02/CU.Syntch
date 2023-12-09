#ifndef AUDIOFORMATINFO_H
#define AUDIOFORMATINFO_H

#include <inttypes.h>
#include <string>
#include <cstdio>

struct audioFormatInfo{
    std::string type;
    uint8_t channels;
    uint8_t bitDepth;
    uint8_t blockAlign;
    uint32_t sampleRate;
    uint32_t sampleSize;
    uint32_t byteRate;
    uint32_t fileLength;
    bool littleEndian;

    unsigned long length();
    unsigned long timeElapsed(uint32_t bytesRead);
    static std::string secondsToString(unsigned long seconds);
    void print();
};

#endif
