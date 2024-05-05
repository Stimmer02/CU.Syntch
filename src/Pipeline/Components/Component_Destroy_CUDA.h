#ifndef COMPONENT_DESTROY_CUDA_H
#define COMPONENT_DESTROY_CUDA_H

#include "AComponent_CUDA.h"

namespace pipeline{
    class Component_Destroy_CUDA: public AComponent_CUDA{
    public:
        Component_Destroy_CUDA(const audioFormatInfo* audioInfo);
        ~Component_Destroy_CUDA();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;


    private:
        static const std::string privateNames[];

        float& subtract = settings.values[0];
    };
}

#endif
