#ifndef _MIDIMESSAGEINTERPRETER_H
#define _MIDIMESSAGEINTERPRETER_H

#include "midiEvent.h"
#include "midiSettings.h"
#include "midiHeaderChunk.h"

#include <cstdio>

typedef unsigned long int ulong;

namespace MIDI{
    class MidiMessageInterpreter{
    public:
        char getFileEvent(FILE* stream, midiEvent& event);
        char executeEvent(const midiEvent& event, uchar* buffer[127], midiSettings& settings,  uint timePlacement, const uint& sampleSize, const uint& sampleRate, const midiCheaderChunk& info);
    private:
        char getVariableLengthValue(FILE* stream, uint32_t& out);
        char ignoreSysEx(FILE* stream);
        char getLongerMessage(FILE* stream, midiEvent& event, uint32_t length);
    };
}

#endif
