#ifndef ADVANCEDCOMPONENT_SUM2_CUDA_H
#define ADVANCEDCOMPONENT_SUM2_CUDA_H

#include "AAdvancedComponent_CUDA.h"

namespace pipeline{
    class AdvancedComponent_Sum2_CUDA: public AAdvancedComponent_CUDA{
    public:
        AdvancedComponent_Sum2_CUDA(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer);
        ~AdvancedComponent_Sum2_CUDA();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;

        bool allNeededConnections() override;

    private:
        static const std::string privateNames[];

        float& vol0 = settings.values[0];
        float& vol1 = settings.values[1];
    };
}

#endif
