#ifndef MIDICHUNK_H
#define MIDICHUNK_H

#include <cstdint>
#include <cstdio>

namespace MIDI{
    struct midiChunk{
        char ID[5];
        uint32_t size;
        long dataPosition;
        long lastPosition;

        midiChunk(){
            ID[4] = 0;
        }
    };
}

#endif
