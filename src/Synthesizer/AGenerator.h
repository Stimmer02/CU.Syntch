#ifndef AGENERATOR_H
#define AGENERATOR_H

#include "noteBuffer.h"
#include "settings.h"
#include "DynamicsController.h"
#include <cmath>

namespace synthesizer {
    class AGenerator{
    public:
        virtual ~AGenerator(){};
        void generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings, const double* dynamicsProfile, const double* releaseProfile);
        virtual inline double soundFunction(noteBuffer& noteBuffer) = 0;
    };

    enum generator_type{
        SINE = 0,
        SQUARE = 1,
        SAWTOOTH = 2,
        LAST = SAWTOOTH,
    };
}

#endif
