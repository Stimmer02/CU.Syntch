#ifndef _IGENERATOR_SAWTOOTH_H
#define _IGENERATOR_SAWTOOTH_H

#include "IGenerator.h"

namespace synthesizer{
    class Generator_Sawtooth : public IGenerator{
    public:
        void generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings);
    };
}

#endif
