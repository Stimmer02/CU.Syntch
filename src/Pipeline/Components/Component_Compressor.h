#ifndef COMPONENT_COMPRESSOR_H
#define COMPONENT_COMPRESSOR_H

#include "AComponent.h"
#include <cmath>

namespace pipeline{
    class Component_Compressor: public AComponent{
    public:
        Component_Compressor(const audioFormatInfo* audioInfo);
        ~Component_Compressor();

        void apply(pipelineAudioBuffer* buffer) override;
        void clear() override;
        void defaultSettings() override;
        void set(uint index, float value) override;


    private:
        static const std::string privateNames[];

        float lLevel;
        float rLevel;

        float levelRiseTime;
        float levelDecreaseTime;

        float& threshold = settings.values[0];
        float& ratio = settings.values[1];
        float& step = settings.values[2];
        float& attack = settings.values[3];
        float& release = settings.values[4];
        float& vol = settings.values[5];
    };
}

#endif
