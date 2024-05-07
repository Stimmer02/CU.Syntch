#ifndef COMPONENT_DISTORTION_CUDA_H
#define COMPONENT_DISTORTION_CUDA_H

#include "AComponent_CUDA.h"

namespace pipeline{
    class Component_Distortion_CUDA: public AComponent_CUDA{
    public:
        Component_Distortion_CUDA(const audioFormatInfo* audioInfo);
        ~Component_Distortion_CUDA();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;


    private:
        static const std::string privateNames[];

        float& gain = settings.values[0];
        float& compress = settings.values[1];
        float& symmetry = settings.values[2];
        float& vol = settings.values[3];
    };
}

#endif
