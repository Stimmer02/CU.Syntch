#ifndef IGENERATOR_NOISE1_H
#define IGENERATOR_NOISE1_H

#include "AGenerator.h"

namespace synthesizer{
    class Generator_Noise1 : public AGenerator{
    public:
        float soundFunction(noteBuffer& noteBuffer) override;
    };
}

#endif
