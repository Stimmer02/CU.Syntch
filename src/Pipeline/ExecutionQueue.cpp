#include "ExecutionQueue.h"

using namespace pipeline;

char ExecutionQueue::build(std::vector<AudioBufferQueue*>& componentQueues, AudioBufferQueue*& outputQueue){
    operations.clear();
    connectedSynthIDs.clear();

    operations.push_back(outputQueue);

    if (outputQueue->parentType == pipeline::SYNTH){
        connectedSynthIDs.push_back(outputQueue->getParentID());
    } else {

    }

    return 0;
}

bool ExecutionQueue::valid(){
    if (connectedSynthIDs.empty()){
        return false;
    }
    return true;
}

const std::vector<AudioBufferQueue*>& ExecutionQueue::getQueue(){
    return operations;
}

const std::vector<short>& ExecutionQueue::getConnectedSynthIDs(){
    return connectedSynthIDs;
}
