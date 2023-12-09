#include "Generator_Sine.h"

inline double synthesizer::Generator_Sine::soundFunction(noteBuffer& noteBuffer){
    return sin(noteBuffer.phaze*noteBuffer.multiplier);
}
