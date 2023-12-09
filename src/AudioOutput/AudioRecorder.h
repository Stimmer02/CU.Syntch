#ifndef AUDIORECORDER_H
#define AUDIORECORDER_H

#include "audioBuffer.h"
#include "audioFormatInfo.h"
#include <fstream>

typedef unsigned char uchar;

class AudioRecorder{
public:
    AudioRecorder();
    ~AudioRecorder();

    char init(const audioFormatInfo& info, std::string fileName);
    char init(const audioFormatInfo& info);
    char saveBuffer(const audioBuffer* buffer);
    char closeFile();

private:
    void writeInverted(ulong input, uchar length);

    std::fstream* file;
    uint savedData;
    long fileSizePosition;
    long dataSizePosition;
};

#endif
