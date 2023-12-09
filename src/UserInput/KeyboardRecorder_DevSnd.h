#ifndef KEYBOARDRECORDER_DEVSND_H
#define KEYBOARDRECORDER_DEVSND_H

#include <string>
#include <fstream>
#include <linux/input.h>
#include <cmath>


#include "AKeyboardRecorder.h"
#include "MIDI/MidiMessageInterpreter.h"


class KeyboardRecorder_DevSnd :public AKeyboardRecorder {
    public:
    KeyboardRecorder_DevSnd(const ushort& keyCount);
    ~KeyboardRecorder_DevSnd();
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
    std::ifstream* inputStream;
    std::thread* scannerThread;

    uint sampleSize;
    uint sampleRate;

    MIDI::MidiMessageInterpreter interpreter;
};
#endif
