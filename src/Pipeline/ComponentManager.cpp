#include "ComponentManager.h"

using namespace pipeline;

ComponentManager::ComponentManager(const audioFormatInfo* audioInfo) : audioInfo(audioInfo){

}

ComponentManager::~ComponentManager(){

}

short ComponentManager::addComponent(){
    return 0;
}

// short ComponentManager::addComponent(){
//
//}

char ComponentManager::removeComponent(short componentID){
    return 0;
}

char ComponentManager::applyEffects(AudioBufferQueue** queues, const short& count){
    return 0;
}

void ComponentManager::printTrace(short ID){

}
