#include "Generator_Sawtooth.h"

inline double synthesizer::Generator_Sawtooth::soundFunction(noteBuffer& noteBuffer){
    return (((noteBuffer.phaze)%uint(noteBuffer.multiplier))/noteBuffer.multiplier*2 - 1);
}
