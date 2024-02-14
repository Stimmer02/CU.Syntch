#ifndef KEYBOARDRECORDER_DEVINPUT_H
#define KEYBOARDRECORDER_DEVINPUT_H

#include <string>
#include <fstream>
#include <linux/input.h>
#include <cmath>


#include "AKeyboardRecorder.h"
#include "InputMap.h"


class KeyboardRecorder_DevInput :public AKeyboardRecorder {
    public:
    KeyboardRecorder_DevInput(const ushort& keyCount, InputMap*& keyboardMap);
    ~KeyboardRecorder_DevInput();
    char init(const std::string path, const uint& sampleSize, const uint& sampleFrequency);
    char reInit(const uint& sampleSize, const uint& sampleFrequency);
    char start();
    char stop();
    bool isRunning();

    const ushort keyCount;

    private:
    void scannerThreadFunction();

    bool running;
    std::string path;
    std::fstream* inputStream;
    std::thread* scannerThread;

    uint sampleSize;
    uint sampleRate;

    const InputMap* keyboardMap;
};
#endif
