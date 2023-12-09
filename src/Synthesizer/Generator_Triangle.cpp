#include "Generator_Triangle.h"

inline double synthesizer::Generator_Triangle::soundFunction(noteBuffer& noteBuffer){
    float volume = noteBuffer.phaze/noteBuffer.multiplier;
    uint round = volume;
    volume -= round;
    volume = (int((noteBuffer.phaze)/noteBuffer.multiplier) & 0x1) == 0 ? volume : 1.0 - volume;
    return volume*2 - 1;

}
