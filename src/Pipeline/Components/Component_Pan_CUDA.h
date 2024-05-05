#ifndef COMPONENT_PAN_CUDA_H
#define COMPONENT_PAN_CUDA_H

#include "AComponent_CUDA.h"

namespace pipeline{
    class Component_Pan_CUDA: public AComponent_CUDA{
    public:
        Component_Pan_CUDA(const audioFormatInfo* audioInfo);
        ~Component_Pan_CUDA();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;


    private:
        static const std::string privateNames[];

        float& pan = settings.values[0];
    };
}

#endif
