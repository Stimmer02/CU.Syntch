#ifndef COMPONENT_VOLUME_CUDA_H
#define COMPONENT_VOLUME_CUDA_H

#include "AComponent_CUDA.h"

namespace pipeline{
    class Component_Volume_CUDA: public AComponent_CUDA{
    public:
        Component_Volume_CUDA(const audioFormatInfo* audioInfo);
        ~Component_Volume_CUDA();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;


    private:
        static const std::string privateNames[];

        float& vol = settings.values[0];
    };
}

#endif
