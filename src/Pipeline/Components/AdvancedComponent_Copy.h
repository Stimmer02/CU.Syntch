#ifndef ADVANCEDCOMPONENT_COPY_H
#define ADVANCEDCOMPONENT_COPY_H

#include "AAdvancedComponent.h"

namespace pipeline{
    class AdvancedComponent_Copy: public AAdvancedComponent{
    public:
        AdvancedComponent_Copy(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer);
        ~AdvancedComponent_Copy();

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
