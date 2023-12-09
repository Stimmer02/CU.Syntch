#include "Generator_Noise1.h"

inline double synthesizer::Generator_Noise1::soundFunction(noteBuffer& noteBuffer){
    return sin(noteBuffer.phaze*noteBuffer.multiplier);
}
