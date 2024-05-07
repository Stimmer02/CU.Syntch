#ifndef COMPONENT_ECHO_CUDA_H
#define COMPONENT_ECHO_CUDA_H

#include "AComponent_CUDA.h"

namespace pipeline{
    class Component_Echo_CUDA: public AComponent_CUDA{
    public:
        Component_Echo_CUDA(const audioFormatInfo* audioInfo);
        ~Component_Echo_CUDA();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;
        void set(uint index, float value);

    private:
        static const std::string privateNames[];

        float* d_lMemory;
        float* d_rMemory;

        int currentSample;
        int sampleCount;

        const uint maxDelayTime;

        float& lvol = settings.values[0];
        float& rvol = settings.values[1];
        float& delay = settings.values[2];
        float& fade = settings.values[3];
        float& repeats = settings.values[4];

    };
}

#endif
