#ifndef ADVANCEDCOMPONENT_COPY_CUDA_H
#define ADVANCEDCOMPONENT_COPY_CUDA_H

#include "AAdvancedComponent_CUDA.h"

namespace pipeline{
    class AdvancedComponent_Copy_CUDA: public AAdvancedComponent_CUDA{
    public:
        AdvancedComponent_Copy_CUDA(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer);
        ~AdvancedComponent_Copy_CUDA();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;

        bool allNeededConnections() override;

    private:
        static const std::string privateNames[];

        float& vol = settings.values[0];
    };
}

#endif
