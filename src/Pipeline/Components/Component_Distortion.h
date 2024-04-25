#ifndef COMPONENT_DISTORTION_H
#define COMPONENT_DISTORTION_H

#include "AComponent.h"

namespace pipeline{
    class Component_Distortion: public AComponent{
    public:
        Component_Distortion(const audioFormatInfo* audioInfo);
        ~Component_Distortion();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;


    private:
        static const std::string privateNames[];

        float& gain = settings.values[0];
        float& compress = settings.values[1];
        float& symmetry = settings.values[2];
        float& vol = settings.values[3];
    };
}

#endif
