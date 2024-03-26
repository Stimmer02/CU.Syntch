#ifndef AKEYBOARDRECORDER_H
#define AKEYBOARDRECORDER_H

#include <string>
#include <thread>

#include "IKeyboardDoubleBuffer.h"


class AKeyboardRecorder{
public:
    virtual ~AKeyboardRecorder() = default;
    virtual char init(const std::string path, const uint& sampleSize, const uint& sampleRate) = 0;
    virtual char reInit(const uint& sampleSize, const uint& sampleRate) = 0;
    virtual char start() = 0;
    virtual char stop() = 0;
    virtual bool isRunning() = 0;

    IKeyboardDoubleBuffer* buffer;
};

#endif
