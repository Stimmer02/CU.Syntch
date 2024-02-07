#include "AudioBufferQueue.h"

using namespace pipeline;

AudioBufferQueue::AudioBufferQueue(const ID_type parentType, const uint sampleSize): buffer(sampleSize), parentType(parentType){
    queueLength = 0;
}

AudioBufferQueue::~AudioBufferQueue(){

}

short AudioBufferQueue::getQueueLenth(){
    return queueLength;
}

short AudioBufferQueue::getParentID(){
    return parentID;
}


// queueItem getQueueItem(short index);
// char addQueueItem(const queueItem& queueItem);
// char removeQueueItem(short index);
