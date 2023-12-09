#ifndef MIDIMESSAGEINTERPRETER_H
#define MIDIMESSAGEINTERPRETER_H

#include "midiEvent.h"
#include "midiSettings.h"
#include "midiHeaderChunk.h"

#include <fstream>


typedef unsigned long int ulong;

namespace MIDI{
    class MidiMessageInterpreter{
    public:
        char getFileEvent(std::ifstream* stream, midiEvent& event);
        char getEvent(std::ifstream* stream, midiEvent& event);
        void executeMidiEvent(const midiEvent& event, uchar* buffer[127], uint timePlacement);
        char executeEvent(const midiEvent& event, uchar* buffer[127], midiSettings& settings,  uint timePlacement, const uint& sampleSize, const uint& sampleRate, const midiCheaderChunk& info);
    private:
        char getVariableLengthValue(std::ifstream* stream, uint32_t& out);
        char ignoreSysEx(std::ifstream* stream);
        char getLongerMessage(std::ifstream* stream, midiEvent& event, uint32_t length);
    };
}

#endif
