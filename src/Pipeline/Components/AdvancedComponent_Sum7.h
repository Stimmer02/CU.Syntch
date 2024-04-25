#ifndef ADVANCEDCOMPONENT_SUM7_H
#define ADVANCEDCOMPONENT_SUM7_H

#include "AAdvancedComponent.h"

namespace pipeline{
    class AdvancedComponent_Sum7: public AAdvancedComponent{
    public:
        AdvancedComponent_Sum7(const audioFormatInfo* audioInfo, audioBufferQueue* boundBuffer);
        ~AdvancedComponent_Sum7();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;

        bool allNeededConnections() override;

    private:
        static const std::string privateNames[];

        float& vol0 = settings.values[0];
        float& vol1 = settings.values[1];
        float& vol2 = settings.values[2];
        float& vol3 = settings.values[3];
        float& vol4 = settings.values[4];
        float& vol5 = settings.values[5];
        float& vol6 = settings.values[6];
    };
}

#endif
