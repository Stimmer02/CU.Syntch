#ifndef _AUDIORECORDER_H
#define _AUDIORECORDER_H

#include "audioBuffer.h"
#include "audioFormatInfo.h"
#include <fstream>

typedef unsigned char uchar;

class AudioRecorder{
public:
    ~AudioRecorder();

    char init(const audioFormatInfo& info, std::string fileName);
    char saveBuffer(const audioBuffer* buffer);
    char closeFile();

private:
    void writeInverted(ulong input, uchar length);

    std::fstream* file;
    uint savedData;
    uint fileSizePosition;
    uint dataSizePosition;
};

#endif
