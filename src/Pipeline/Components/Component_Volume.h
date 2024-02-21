#ifndef COMPONENT_VOLUME_H
#define COMPONENT_VOLUME_H

#include "AComponent.h"

namespace pipeline{
    class Component_Volume: public AComponent{
    public:
        Component_Volume(const audioFormatInfo* audioInfo);
        ~Component_Volume();

        void apply(pipelineAudioBuffer* buffer) override;
        void clear() override;
        void defaultSettings() override;


    private:
        static const std::string privateNames[];
    };
}

#endif
