#ifndef KEYBOARDRECORDER_DEVSND_H
#define KEYBOARDRECORDER_DEVSND_H

#include <string>
#include <fstream>
#include <linux/input.h>
#include <cmath>


#include "AKeyboardRecorder.h"
#include "KeyboardDoubleBuffer.h"
#include "MIDI/MidiMessageInterpreter.h"


class KeyboardRecorder_DevSnd :public AKeyboardRecorder {
    public:
    KeyboardRecorder_DevSnd(const ushort& keyCount);
    ~KeyboardRecorder_DevSnd() override;
    char init(const std::string path, const uint& sampleSize, const uint& sampleFrequency) override;
    char reInit(const uint& sampleSize, const uint& sampleFrequency) override;
    char start() override;
    char stop() override;
    bool isRunning() override;

    const ushort keyCount;

    private:
    void scannerThreadFunction();

    bool running;
    std::string path;
    std::ifstream* inputStream;
    std::thread* scannerThread;

    uint sampleSize;
    uint sampleRate;

    MIDI::MidiMessageInterpreter interpreter;
};
#endif
