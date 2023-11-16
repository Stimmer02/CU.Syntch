#ifndef _IGENERATOR_TRIANGLE_H
#define _IGENERATOR_TRIANGLE_H

#include "IGenerator.h"

namespace synthesizer{
    class Generator_Triangle : public IGenerator{
    public:
        void generate(noteBuffer& noteBuffer, const settings& settings);
    };
}

#endif
