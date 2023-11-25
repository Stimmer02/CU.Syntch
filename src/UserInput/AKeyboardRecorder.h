#ifndef _AKEYBOARDRECORDER_H
#define _AKEYBOARDRECORDER_H

#include <string>
#include <thread>

#include "KeyboardDoubleBuffer.h"
#include "InputMap.h"


class AKeyboardRecorder{
public:
    virtual ~AKeyboardRecorder(){};
    virtual char init(const std::string path, const uint& sampleSize, const uint& sampleRate, InputMap* keyboardMap) = 0;
    virtual char reInit(const uint& sampleSize, const uint& sampleFrequency) = 0;
    virtual char start() = 0;
    virtual char stop() = 0;
    virtual bool isRunning() = 0;

    KeyboardDoubleBuffer* buffer;
protected:
    InputMap* keyboardMap;
};

#endif
