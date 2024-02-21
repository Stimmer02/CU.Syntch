#include "ExecutionQueue.h"


using namespace pipeline;

bool compareFunction(audioBufferQueue*& a, audioBufferQueue*& b){
    return a->getParentID() == b->getParentID() && a->parentType == b->parentType;
}

void ExecutionQueue::build(std::vector<audioBufferQueue*>& componentQueues, audioBufferQueue*& outputQueue, IDManager<AComponent, short>& components){
    operations.clear();
    connectedSynthIDs.clear();
    invalidAdvancedComponents.clear();



    if (outputQueue->parentType == pipeline::SYNTH){
        operations.push_back(outputQueue);
        connectedSynthIDs.push_back(outputQueue->getParentID());
    } else {
        ShiftBuffer<audioBufferQueue*> toBuild(componentQueues.size(), &compareFunction);
        audioBufferQueue** builded = new audioBufferQueue*[componentQueues.size()];
        short buildedCount = 0;

        toBuild.put(outputQueue);

        while (toBuild.leftInBuffer() > 0){
            builded[buildedCount] = toBuild.get();
            bool unique = true;
            for (short i = 0; i < buildedCount; i++){
                if (builded[i]->getParentID() == builded[buildedCount]->getParentID() && builded[i]->parentType == builded[buildedCount]->parentType){
                    unique = false;
                    break;
                }
            }
            if (unique == false){
                continue;
            }
            operations.push_back(builded[buildedCount]);
            if (builded[buildedCount]->parentType == pipeline::SYNTH){
                connectedSynthIDs.push_back(builded[buildedCount]->getParentID());
            } else {
                AAdvancedComponent* advComp = reinterpret_cast<AAdvancedComponent*>(components.getElement(builded[buildedCount]->getParentID()));
                if (advComp->allNeededConnections() == false){
                    invalidAdvancedComponents.push_back(builded[buildedCount]->getParentID());
                }
                for (uint i = 0; i < advComp->maxConnections; i++){
                    audioBufferQueue* connection = advComp->getConnection(i);
                    if (connection != nullptr){
                        toBuild.putIfUnique(connection);
                    }
                }
            }
            buildedCount++;
        }
        delete[] builded;
    }
}

char ExecutionQueue::error(){
    if (connectedSynthIDs.empty()){
        return -1;
    }
    if (invalidAdvancedComponents.empty() == false){
        return -2;
    }

    return 0;
}

const std::vector<audioBufferQueue*>& ExecutionQueue::getQueue(){
    return operations;
}

const std::vector<short>& ExecutionQueue::getConnectedSynthIDs(){
    return connectedSynthIDs;
}

const std::vector<short>& ExecutionQueue::getInvalidAdvCompIDs(){
    return invalidAdvancedComponents;
}


