#ifndef COMPONENT_DESTROY_H
#define COMPONENT_DESTROY_H

#include "AComponent.h"

namespace pipeline{
    class Component_Destroy: public AComponent{
    public:
        Component_Destroy(const audioFormatInfo* audioInfo);
        ~Component_Destroy();

        void apply(pipelineAudioBuffer_CUDA* buffer) override;
        void clear() override;
        void defaultSettings() override;


    private:
        static const std::string privateNames[];

        float& subtract = settings.values[0];
    };
}

#endif
