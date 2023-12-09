#ifndef IGENERATOR_SAWTOOTH_H
#define IGENERATOR_SAWTOOTH_H

#include "IGenerator.h"

namespace synthesizer{
    class Generator_Sawtooth : public IGenerator{
    public:
        void generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings, const double* dynamicsProfile, const double* releaseProfile);
    };
}

#endif
