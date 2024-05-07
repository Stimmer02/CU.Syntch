#ifndef ACOMPONENT_H
#define ACOMPONENT_H

#include "../pipelineAudioBuffer_CUDA.h"
#include "../../AudioOutput/audioFormatInfo.h"
#include "../audioBufferQueue.h"
#include "componentSettings_CUDA.h"

#include <string>
#include <vector>

#define COMPONENT_BLOCK_SIZE 256

namespace pipeline{
    enum component_type{
        COMP_INVALID,
        COMP_VOLUME,
        COMP_PAN,
        COMP_ECHO,
        COMP_DISTORION,
        COMP_COMPRESSOR,
        COMP_DESTROY,
    };

    class AComponent_CUDA{
    public:
        AComponent_CUDA(const audioFormatInfo* audioInfo, uint settingCount, const std::string* settingNames, component_type type);
        virtual ~AComponent_CUDA() = default;

        virtual void apply(pipelineAudioBuffer_CUDA* buffer) = 0;
        virtual void clear() = 0;
        virtual void defaultSettings() = 0;

        virtual void set(uint index, float value);
        const componentSettings_CUDA* getSettings();

        audioBufferQueue* includedIn;

        const component_type type;

    protected:
        componentSettings_CUDA settings;
        const audioFormatInfo* audioInfo;
    };
}

#endif
