#ifndef DYNAMICSCONTROLLER_CUDA_H
#define DYNAMICSCONTROLLER_CUDA_H

#include "settings_CUDA.h"
#include <vector>
#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cuda.h>

typedef unsigned long int ulong;

namespace synthesizer{
    class DynamicsController_CUDA{
    public:
        DynamicsController_CUDA();
        ~DynamicsController_CUDA();

        void calculateDynamicsProfile(settings_CUDA* d_settings, settings_CUDA& settings);
        void calculateReleaseProfile(settings_CUDA* d_settings, settings_CUDA& settings, float rawReleaseDuration);
        const float* getDynamicsProfile();
        const float* getReleaseProfile();
        const uint* getReleaseToAttackIndexMap();
        uint getDynamicsProfileLength();
        uint getReleaseProfileLength();

        //do not use for release dynamics, for this case use calculateReleaseProfile
        void setDynamics(dynamics_CUDA* d_dynamics, dynamics_CUDA& dynamics, const float& raw, const uint& sampleRate);

    private:
        void allocateDynamicsMemory(const settings_CUDA& settings);
        void allocateReleaseMemory(const uint& newReleaseDuration);

        float* d_noteDynamicsProfile;
        uint dynamicsDuration;
        uint dynamicsAllocated;

        float* d_noteReleaseProfile;
        uint* d_releaseToAttackIndexMap;
        uint releaseDuration;
        uint releaseAllocated;
    };
}

#endif
