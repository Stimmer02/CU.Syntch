#ifndef _IGENERATOR_H
#define _IGENERATOR_H

#include "noteBuffer.h"
#include "settings.h"
#include <cmath>

namespace synthesizer {
    class IGenerator{
    public:
        virtual ~IGenerator(){};
        virtual void generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings) = 0;
    };

    enum generator_type{
        SINE,
        SQUARE,
        TRIANGLE,
    };
}

#endif
