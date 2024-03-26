#include "audioBufferQueue.h"

using namespace pipeline;

audioBufferQueue::audioBufferQueue(const ID_type parentType, const uint sampleSize): buffer(sampleSize), parentType(parentType){
    parentID = -2;
}

audioBufferQueue::~audioBufferQueue(){

}


short audioBufferQueue::getParentID(){
    return parentID;
}


// queueItem getQueueItem(short index);
// char addQueueItem(const queueItem& queueItem);
// char removeQueueItem(short index);
