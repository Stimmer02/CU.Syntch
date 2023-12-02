#ifndef _AUDIORECORDER_H
#define _AUDIORECORDER_H

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
    char initFile(const audioFormatInfo& info);
    void writeInverted(ulong input, uchar length);

    FILE* file;
    uint savedData;
    fpos_t fileSizePosition;
    fpos_t dataSizePosition;
};

#endif
