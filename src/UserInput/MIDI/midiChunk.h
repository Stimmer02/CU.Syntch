#ifndef _MIDICHUNK_H
#define _MIDICHUNK_H

#include <cstdint>
#include <cstdio>

namespace MIDI{
    struct midiChunk{
        char ID[5];
        uint32_t size;
        fpos_t dataPosition;
        fpos_t lastPosition;

        midiChunk(){
            ID[4] = 0;
        }
    };
}

#endif
