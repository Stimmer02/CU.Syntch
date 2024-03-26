#ifndef AUDIOFORMATINFO_H
#define AUDIOFORMATINFO_H

#include <inttypes.h>
#include <string>
#include <cstdio>

#define DEFAULT_CHANNELS 2
#define DEFAULT_BIT_DEPTH 16
#define DEFAULT_SAMPLE_RATE 44100
#define DEFAULT_SAMPLE_SIZE 1024
#define DEFAULT_LITTLE_ENDIAN true

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

    audioFormatInfo() : type(""), channels(DEFAULT_CHANNELS), bitDepth(DEFAULT_BIT_DEPTH), blockAlign(0),
                        sampleRate(DEFAULT_SAMPLE_RATE), sampleSize(DEFAULT_SAMPLE_SIZE),
                        fileLength(0), littleEndian(DEFAULT_LITTLE_ENDIAN) {
                            byteRate = sampleRate * channels * bitDepth/8;
                            blockAlign = channels * bitDepth/8;
                        }

    unsigned long length();
    unsigned long timeElapsed(uint32_t bytesRead);
    static std::string secondsToString(unsigned long seconds);
    void print();
};

#endif
