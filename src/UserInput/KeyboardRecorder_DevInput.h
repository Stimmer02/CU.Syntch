#ifndef _KEYBOARDRECORDER_DEVINPUT_H
#define _KEYBOARDRECORDER_DEVINPUT_H

#include <string>
#include <fstream>
#include <linux/input.h>
#include <cmath>


#include "AKeyboardRecorder.h"


class KeyboardRecorder_DevInput :public AKeyboardRecorder {
    public:
    KeyboardRecorder_DevInput(const ushort& keyCount);
    ~KeyboardRecorder_DevInput();
    char init(const std::string path, const uint& sampleSize, const uint& sampleFrequency, InputMap* keyboardMap);
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
};
#endif
