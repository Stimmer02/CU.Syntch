#ifndef COMPONENTMANAGER_H
#define COMPONENTMANAGER_H

#include "IDManager.h"
#include "../AudioOutput/audioFormatInfo.h"
#include "audioBufferQueue.h"

#include "Components/AComponent.h"
#include "Components/Component_Volume.h"
#include "Components/Component_Pan.h"
#include "Components/Component_Echo.h"
#include "Components/Component_Distortion.h"
#include "Components/Component_Compressor.h"
#include "Components/Component_Destroy.h"


#include "Components/AAdvancedComponent.h"
#include "Components/AdvancedComponent_Sum2.h"

#include <set>


namespace pipeline{
    class ComponentManager{
    public:
        ComponentManager(const audioFormatInfo* audioInfo);
        ~ComponentManager();

        short addComponent(component_type type);
        short addComponent(advanced_component_type type, audioBufferQueue* boundBuffer);
        // char removeComponent(short componentID);
        char addComonentToQueue(short componentID, short queueParentID, ID_type queueParentType);
        char applyEffects(audioBufferQueue* queue);
        void printTrace(short ID);
        AAdvancedComponent* getAdvancedComponent(short componentID);

        const audioFormatInfo* audioInfo;

        IDManager<AComponent, short> components;
        std::set<short> advancedIDs;
    };
}

#endif
