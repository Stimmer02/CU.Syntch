#ifndef _IGENERATOR_H
#define _IGENERATOR_H

#include "noteBuffer.h"
#include "settings.h"
#include "DynamicsController.h"
#include <cmath>

namespace synthesizer {
    class IGenerator{
    public:
        virtual ~IGenerator(){};
        virtual void generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings, const double* dynamicsProfile, const double* releaseProfile) = 0;
    };

    enum generator_type{
        SINE = 0,
        SQUARE = 1,
        SAWTOOTH = 2,
        LAST = SAWTOOTH,
    };
}

#endif
