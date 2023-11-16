#ifndef _IGENERATOR_SINE_H
#define _IGENERATOR_SINE_H

#include "IGenerator.h"

namespace synthesizer{
    class Generator_Sine : public IGenerator{
    public:
        void generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings);
    };
}

#endif
