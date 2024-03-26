#ifndef KEYBOARDRECORDER_MIDIFILE_H
#define KEYBOARDRECORDER_MIDIFILE_H

#include <string>

#include "../AKeyboardRecorder.h"
#include "../KeyboardDoubleBuffer_Empty.h"
#include "KeyboardDoubleBuffer_MidiFile.h"
#include "IMidiFileReaderObserver.h"

namespace MIDI{
    class KeyboardRecorder_MidiFile :public AKeyboardRecorder, IMidiFileReaderObserver{
    public:
        KeyboardRecorder_MidiFile();
        ~KeyboardRecorder_MidiFile() override;
        char init(std::string path, const uint& sampleSize, const uint& sampleRate) override;
        char init(const uint& sampleSize, const uint& sampleRate);
        char reInit(const uint& sampleSize, const uint& sampleFrequency) override;
        char reInitFile(std::string path);
        char start() override;
        char stop() override;
        bool isRunning() override;

        char play();
        char pause();
        bool isPlaying();
        char revind();
        bool isReady();
        std::string getPath();

        void setPlayCounter(int* playCounter);

    private:

        bool running; //mostly for interface compatibility
        bool playing;

        uint sampleSize;
        uint sampleRate;

        std::string path;

        KeyboardDoubleBuffer_Empty* emptyBuffer;
        KeyboardDoubleBuffer_MidiFile* midiBuffer;

        int* playCounter;

        void notifyFileEnd() override;
    };
}


#endif
