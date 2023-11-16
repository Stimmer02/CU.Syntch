#ifndef _SETTINGS_H
#define _SETTINGS_H

#include "dynamics.h"

typedef unsigned short int ushort;

namespace synthesizer{
    struct settings{
        ushort keyCount;
        uint sampleSize;
        uint sampleRate;
        uint maxValue;

        char pitch;
        float volume;

        dynamics attack;
        dynamics sustain;
        dynamics fade;
        dynamics release;
    };
}
#endif
