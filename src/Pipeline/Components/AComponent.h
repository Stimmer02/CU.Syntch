#ifndef ACOMPONENT_H
#define ACOMPONENT_H

#include "../pipelineAudioBuffer.h"
#include "../../AudioOutput/audioFormatInfo.h"
#include "../audioBufferQueue.h"
#include "componentSettings.h"

#include <string>
#include <vector>

namespace pipeline{
    class AComponent{
    public:
        AComponent(const audioFormatInfo* audioInfo, uint settingCount, const std::string* settingNames);
        virtual ~AComponent() = default;

        virtual void apply(pipelineAudioBuffer* buffer) = 0;
        virtual void clear() = 0;
        virtual void defaultSettings() = 0;

        void set(uint index, float value);
        const componentSettings* getSettings();

        audioBufferQueue* includedIn;

    protected:
        componentSettings settings;
        const audioFormatInfo* audioInfo;

        //parentID
    };
}

#endif
