#include "Generator_Noise1.h"

inline float synthesizer::Generator_Noise1::soundFunction(noteBuffer& noteBuffer){
    return sin(noteBuffer.phaze*noteBuffer.multiplier);
}
