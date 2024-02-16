#include "ComponentManager.h"


using namespace pipeline;

ComponentManager::ComponentManager(const audioFormatInfo* audioInfo) : audioInfo(audioInfo){}

ComponentManager::~ComponentManager(){}

short ComponentManager::addComponent(component_type type){
    AComponent* newComponent;
    switch (type) {
        case COMP_INVALID:
            return -1;
        case COMP_VOLUME:
            newComponent = new Component_Volume(audioInfo);
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
