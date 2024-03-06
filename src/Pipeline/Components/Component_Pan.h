#ifndef COMPONENT_PAN_H
#define COMPONENT_PAN_H

#include "AComponent.h"

namespace pipeline{
    class Component_Pan: public AComponent{
    public:
        Component_Pan(const audioFormatInfo* audioInfo);
        ~Component_Pan();

        void apply(pipelineAudioBuffer* buffer) override;
        void clear() override;
        void defaultSettings() override;


    private:
        static const std::string privateNames[];

        float& pan = settings.values[0];
    };
}

#endif
