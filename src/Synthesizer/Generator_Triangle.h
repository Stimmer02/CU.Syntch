#ifndef IGENERATOR_TRIANGLE_H
#define IGENERATOR_TRIANGLE_H

#include "AGenerator.h"

namespace synthesizer{
    class Generator_Triangle : public AGenerator{
    public:
        float soundFunction(noteBuffer& noteBuffer) override;
    };
}

#endif
