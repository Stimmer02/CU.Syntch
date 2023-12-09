#ifndef IGENERATOR_SQUARE_H
#define IGENERATOR_SQUARE_H

#include "AGenerator.h"

namespace synthesizer{
    class Generator_Square : public AGenerator{
    public:
        float soundFunction(noteBuffer& noteBuffer) override;
    };
}

#endif
