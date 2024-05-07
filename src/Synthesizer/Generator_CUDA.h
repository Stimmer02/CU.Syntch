#ifndef GENERATOR_CUDA_H
#define GENERATOR_CUDA_H

#include "NoteBufferHandler_CUDA.h"
#include "settings_CUDA.h"
#include "DynamicsController_CUDA.h"
#include <cmath>

namespace synthesizer {
    enum generator_type{
        SINE = 0,
        SQUARE = 1,
        SAWTOOTH = 2,
        TRIANGLE = 3,
        NOISE1 = 4,
        LAST = NOISE1,
        INVALID_GEN
    };

    class Generator_CUDA{
    public:
        Generator_CUDA(const settings_CUDA& settings);
        ~Generator_CUDA();
        void generate(noteBuffer_CUDA* d_noteBuffer, const uchar* d_keyState, const settings_CUDA* d_settings, const settings_CUDA& settings, const float* d_dynamicsProfile, const float* d_releaseProfile, const uint* d_releaseToAttackIndexMap);

        void setGenerator(generator_type type);
        generator_type getGeneratorType();

        typedef float (*soundFunctionPointer)(const float phaze, const float multiplier);
    private:

        soundFunctionPointer* d_soundFunctions; 
        generator_type currentGeneratorType;

        uint* d_phazeWorkArr;
        uint* d_pressSamplessPassedWorkArr;
        uint* d_releaseSamplesPassedWorkArr;
        float* d_velocityWorkArr;
    };
}

#endif
