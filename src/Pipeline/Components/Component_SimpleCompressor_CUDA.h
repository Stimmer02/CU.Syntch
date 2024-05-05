#ifndef COMPONENT_SIMPLECOMPRESSOR_CUDA_H
#define COMPONENT_SIMPLECOMPRESSOR_CUDA_H

#include "AComponent_CUDA.h"
#include <cmath>

namespace pipeline{
    class Component_SimpleCompressor_CUDA: public AComponent_CUDA{
    public:
        Component_SimpleCompressor_CUDA(const audioFormatInfo* audioInfo);
        ~Component_SimpleCompressor_CUDA();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;
        void set(uint index, float value) override;


    private:
        static const std::string privateNames[];

        float& threshold = settings.values[0];
        float& ratio = settings.values[1];
        float& vol = settings.values[2];
    };
}

#endif
