#ifndef COMPONENTMANAGER_H
#define COMPONENTMANAGER_H

#include "../AudioOutput/audioFormatInfo.h"
#include "AudioBufferQueue.h"
#include "IDManager.h"

namespace pipeline{
    class ComponentManager{
    public:
        ComponentManager(const audioFormatInfo* audioInfo);
        ~ComponentManager();

        short addComponent();
        // short addComponent();
        char removeComponent(short componentID);
        char applyEffects(AudioBufferQueue** queues, const short& count);
        void printTrace(short ID);

        const audioFormatInfo* audioInfo;

    private:
        short componentCount;
        // APipelineComponent** components;
    };
}

#endif
