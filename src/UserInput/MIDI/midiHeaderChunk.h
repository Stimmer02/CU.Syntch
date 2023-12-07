#ifndef _MIDIHEADERCHUNK_H
#define _MIDIHEADERCHUNK_H

#include "midiChunk.h"

namespace MIDI{
    struct midiCheaderChunk : public midiChunk{
        uint16_t formatType;
        uint16_t trackCount;
        uint16_t timeDivision;
    };
}

#endif
