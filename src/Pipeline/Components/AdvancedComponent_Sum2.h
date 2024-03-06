#ifndef ADVANCEDCOMPONENT_SUM2_H
#define ADVANCEDCOMPONENT_SUM2_H

#include "AAdvancedComponent.h"

namespace pipeline{
    class AdvancedComponent_Sum2: public AAdvancedComponent{
    public:
        AdvancedComponent_Sum2(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer);
        ~AdvancedComponent_Sum2();

        void apply(pipelineAudioBuffer* buffer) override;
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
