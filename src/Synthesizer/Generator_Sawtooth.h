#ifndef IGENERATOR_SAWTOOTH_H
#define IGENERATOR_SAWTOOTH_H

#include "AGenerator.h"

namespace synthesizer{
    class Generator_Sawtooth : public AGenerator{
    public:
        float soundFunction(noteBuffer& noteBuffer) override;
    };
}

#endif
