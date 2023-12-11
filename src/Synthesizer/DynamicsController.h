#ifndef DYNAMICSCONTROLLER_H
#define DYNAMICSCONTROLLER_H

#include "dynamics.h"
#include "settings.h"
#include <vector>
#include <cstdlib>

typedef unsigned long int ulong;

namespace synthesizer{
    class DynamicsController{
    public:
        void calculateDynamicsProfile(settings& settings);
        void calculateReleaseProfile(settings& settings, float rawReleaseDuration);
        const float* getDynamicsProfile();
        const float* getReleaseProfile();
        uint getDynamicsProfileLength();
        uint getReleaseProfileLength();

    private:
        std::vector<float> noteDynamicsProfile;
        std::vector<float> noteReleaseProfile;
    };
}

#endif
