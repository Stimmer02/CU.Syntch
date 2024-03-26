#ifndef KEYBOARDDOUBLEBUFFER_MIDIFILE_H
#define KEYBOARDDOUBLEBUFFER_MIDIFILE_H

#include "../IKeyboardDoubleBuffer.h"
#include "MidiFileReader.h"

#include <chrono>


namespace MIDI{
    class KeyboardDoubleBuffer_MidiFile: public IKeyboardDoubleBuffer{

    public:
        KeyboardDoubleBuffer_MidiFile(const std::string path, const uint& sampleSize, const uint& sampleRate);
        ~KeyboardDoubleBuffer_MidiFile() override;
        uchar** getInactiveBuffer() override;
        uchar** getActiveBuffer() override;
        void swapActiveBuffer() override;
        void clearInactiveBuffer() override;
        long getActivationTimestamp() override;

        ushort getKeyCount() override;
        uint getSampleSize() override;

        MidiFileReader fileReader;

    private:
        static const ushort keyCount = 127;
        const uint sampleSize;

        long activationTimestamp;

    };
}
#endif
