#ifndef _DYNAMICSCONTROLLER_H
#define _DYNAMICSCONTROLLER_H

#include "dynamics.h"
#include "settings.h"
#include <vector>
#include <cstdlib>

typedef unsigned long int ulong;

namespace synthesizer{
    class DynamicsController{
    public:
        void calculateDynamicsProfile(settings& settings);
        void calculateReleaseProfile(settings& settings);
        const double* getDynamicsProfile();
        const double* getReleaseProfile();
        uint getDynamicsProfileLength();
        uint getReleaseProfileLength();

    private:
        std::vector<double> noteDynamicsProfile;
        std::vector<double> noteReleaseProfile;
    };
}

#endif
