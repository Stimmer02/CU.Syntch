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

        dynamics attack;  //how long signal rises to maximum volume
        dynamics sustain; //how long signal stays at maximum volume
        dynamics fade;    //how long signal fades after sustain part even if note is still being played
        double fadeTo;    //value fade strives to achieve
        double rawFadeTo; //raw value set by user
        dynamics release; //how long signal is still being faded down after it is not being played anymore

        uint dynamicsDuration; //ovarall duration of attack, sustain and release

        double stereoMix;
    };
}
#endif
