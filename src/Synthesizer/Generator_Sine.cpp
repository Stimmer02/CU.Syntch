#include "Generator_Sine.h"

inline float synthesizer::Generator_Sine::soundFunction(noteBuffer& noteBuffer){
    return sin(noteBuffer.phaze*noteBuffer.multiplier);
}
