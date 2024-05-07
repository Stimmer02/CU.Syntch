#ifndef SETTINGS_CUDA_H
#define SETTINGS_CUDA_H

#include "dynamics_CUDA.h"

typedef unsigned short int ushort;

namespace synthesizer{
    struct settings_CUDA{
        ushort keyCount;
        uint sampleSize;
        uint sampleRate;
        uint maxValue;

        char pitch;
        float volume;

        dynamics_CUDA attack;  //how long signal rises to maximum volume
        dynamics_CUDA sustain; //how long signal stays at maximum volume
        dynamics_CUDA fade;    //how long signal fades after sustain part even if note is still being played
        float fadeTo;    //value fade strives to achieve
        float rawFadeTo; //raw value set by user
        dynamics_CUDA release; //how long signal is still being faded down after it is not being played anymore

        uint dynamicsDuration; //ovarall duration of attack, sustain and release

        float stereoMix;
    };
}
#endif
