#ifndef COMPONENTMANAGER_H
#define COMPONENTMANAGER_H

#include "IDManager.h"
#include "../AudioOutput/audioFormatInfo.h"
#include "audioBufferQueue.h"

#include "Components/AComponent_CUDA.h"
#include "Components/Component_Volume_CUDA.h"
#include "Components/Component_Pan_CUDA.h"
#include "Components/Component_Echo_CUDA.h"
#include "Components/Component_Distortion_CUDA.h"
#include "Components/Component_SimpleCompressor_CUDA.h"
#include "Components/Component_Destroy_CUDA.h"


#include "Components/AAdvancedComponent_CUDA.h"
#include "Components/AdvancedComponent_Sum2_CUDA.h"
#include "Components/AdvancedComponent_Sum7_CUDA.h"
#include "Components/AdvancedComponent_Copy_CUDA.h"

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
        void printTrace(short ID);//TODO
        AAdvancedComponent_CUDA* getAdvancedComponent(short componentID);
        void clearBuffers();

        const audioFormatInfo* audioInfo;

        IDManager<AComponent_CUDA, short> components;
        std::set<short> advancedIDs;
    };
}

#endif
