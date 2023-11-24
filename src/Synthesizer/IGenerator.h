#ifndef _IGENERATOR_H
#define _IGENERATOR_H

#include "noteBuffer.h"
#include "settings.h"
#include <cmath>
// #include <math.h>

namespace synthesizer {
    class IGenerator{
    public:
        virtual ~IGenerator(){};
        virtual void generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings) = 0;
    };

    enum generator_type{
        SINE = 0,
        SQUARE = 1,
        TRIANGLE = 2,
        LAST = TRIANGLE,
    };
}

#endif
