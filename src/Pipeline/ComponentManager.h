#ifndef COMPONENTMANAGER_H
#define COMPONENTMANAGER_H

#include "IDManager.h"
#include "../AudioOutput/audioFormatInfo.h"
#include "audioBufferQueue.h"
#include "Components/AComponent.h"
#include "Components/Component_Volume.h"


#include <thread>


namespace pipeline{
    enum component_type{
        COMP_INVALID,
        COMP_VOLUME,
    };

    class ComponentManager{
    public:
        ComponentManager(const audioFormatInfo* audioInfo);
        ~ComponentManager();

        short addComponent(component_type type);
        // char removeComponent(short componentID);
        char addComonentToQueue(short componentID, short queueParentID, ID_type queueParentType);
        char applyEffects(audioBufferQueue* queue);
        void printTrace(short ID);

        const audioFormatInfo* audioInfo;

        IDManager<AComponent, short> components;
    };
}

#endif
