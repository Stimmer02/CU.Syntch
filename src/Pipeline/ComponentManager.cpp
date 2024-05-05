#include "ComponentManager.h"


using namespace pipeline;

ComponentManager::ComponentManager(const audioFormatInfo* audioInfo) : audioInfo(audioInfo){}

ComponentManager::~ComponentManager(){}

short ComponentManager::addComponent(component_type type){
    AComponent_CUDA* newComponent;
    switch (type) {
        case COMP_INVALID:
            return -1;
        case COMP_VOLUME:
            newComponent = new Component_Volume_CUDA(audioInfo);
            break;
        case COMP_PAN:
            newComponent = new Component_Pan_CUDA(audioInfo);
            break;
        case COMP_ECHO:
            newComponent = new Component_Echo_CUDA(audioInfo);
            break;
        case COMP_DISTORION:
            newComponent = new Component_Distortion_CUDA(audioInfo);
            break;
        case COMP_COMPRESSOR:
            newComponent = new Component_SimpleCompressor_CUDA(audioInfo);
            break;
        case COMP_DESTROY:
            newComponent = new Component_Destroy_CUDA(audioInfo);
            break;
        }

    return components.add(newComponent);
}

short ComponentManager::addComponent(advanced_component_type type, audioBufferQueue* boundBuffer){
    AComponent_CUDA* newComponent;
    switch (type) {
        case ACOMP_INVALID:
            return -1;
        case ACOMP_SUM2:
            newComponent = new AdvancedComponent_Sum2_CUDA(audioInfo, boundBuffer);
            break;
        case ACOMP_SUM7:
            newComponent = new AdvancedComponent_Sum7_CUDA(audioInfo, boundBuffer);
            break;
        case ACOMP_COPY:
            newComponent = new AdvancedComponent_Copy_CUDA(audioInfo, boundBuffer);
            break;
    }

    return components.add(newComponent);
}


char ComponentManager::applyEffects(audioBufferQueue* queue){
    for (uint i = 0; i < queue->componentIDQueue.size(); i++){
        components.getElement(queue->componentIDQueue.at(i))->apply(&queue->buffer);
    }
    return 0;
}

void ComponentManager::printTrace(short ID){

}

AAdvancedComponent_CUDA* ComponentManager::getAdvancedComponent(short componentID){
    if (advancedIDs.find(componentID) == advancedIDs.end()){
        return nullptr;
    }

    return reinterpret_cast<AAdvancedComponent_CUDA*>(components.getElement(componentID));
}

void ComponentManager::clearBuffers(){
    components.doForAll(&AComponent_CUDA::clear);
}

