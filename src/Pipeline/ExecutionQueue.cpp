#include "ExecutionQueue.h"


using namespace pipeline;

bool compareFunction(audioBufferQueue*& a, audioBufferQueue*& b){
    return a->getParentID() == b->getParentID() && a->parentType == b->parentType;
}

void ExecutionQueue::build(std::vector<audioBufferQueue*>& componentQueues, audioBufferQueue*& outputQueue, IDManager<AComponent_CUDA, short>& components){
    operations.clear();
    connectedSynthIDs.clear();
    invalidAdvancedComponents.clear();



    if (outputQueue->parentType == pipeline::SYNTH){
        operations.push_back(outputQueue);
        connectedSynthIDs.push_back(outputQueue->getParentID());
    } else {
        UniqueShiftBuffer<audioBufferQueue*> toBuild(componentQueues.size(), &compareFunction);
        audioBufferQueue* currentElement;

        toBuild.putOrMoveToEnd(outputQueue);

        while (toBuild.leftInBuffer() > 0){
            currentElement = toBuild.get();

            bool uniqueElement = true;
            for (uint i = 0; i < operations.size(); i++){
                if (operations.at(i)->getParentID() == currentElement->getParentID() && operations.at(i)->parentType == currentElement->parentType){
                    uniqueElement = false;
                    operations.erase(operations.cbegin() + i);
                    break;
                }
            }
            operations.push_back(currentElement);

            if (currentElement->parentType == pipeline::SYNTH){
                if (uniqueElement){
                    connectedSynthIDs.push_back(currentElement->getParentID());
                }
            } else {
                AAdvancedComponent_CUDA* advComp = reinterpret_cast<AAdvancedComponent_CUDA*>(components.getElement(currentElement->getParentID()));
                if (advComp->allNeededConnections() == false && uniqueElement){
                    invalidAdvancedComponents.push_back(currentElement->getParentID());
                }
                for (int i = 0; i < advComp->maxConnections; i++){
                    audioBufferQueue* connection = advComp->getConnection(i);
                    if (connection != nullptr){
                        toBuild.putOrMoveToEnd(connection);
                    }
                }
            }
        }
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


