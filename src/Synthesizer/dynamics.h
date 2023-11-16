#ifndef _DYNAMICS_H
#define _DYNAMICS_H

typedef unsigned int uint;

namespace synthesizer{
    struct dynamics{
        float raw;
        uint duration;

        void set(const float& raw, const uint& sampleRate){
            this->raw = raw;
            this->duration = sampleRate*raw;
        }
    };
}
#endif
