#include "Generator_Sawtooth.h"

inline float synthesizer::Generator_Sawtooth::soundFunction(noteBuffer& noteBuffer){
    float volume = noteBuffer.phaze/noteBuffer.multiplier;
    uint round = volume;
    volume -= round;
    return volume*2 - 1;
}
