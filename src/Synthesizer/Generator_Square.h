#ifndef _IGENERATOR_SQUARE_H
#define _IGENERATOR_SQUARE_H

#include "IGenerator.h"

namespace synthesizer{
    class Generator_Square : public IGenerator{
    public:
        void generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings, const double* dynamicsProfile, const double* releaseProfile);
    };
}

#endif
