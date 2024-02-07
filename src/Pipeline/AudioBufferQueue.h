#ifndef AUDIOBUFFERQUEUE_H
#define AUDIOBUFFERQUEUE_H

#include "pipelineAudioBuffer.h"
#include "IDManager.h"
#include <string>



namespace pipeline{

    class AudioBufferQueue{
    public:
        AudioBufferQueue(const ID_type parentType, const uint sampleSize);
        ~AudioBufferQueue();

        short getQueueLenth();
        // queueItem getQueueItem(short index);
        // char addQueueItem(const queueItem& queueItem);
        short getParentID();
        char removeQueueItem(short index);

        pipelineAudioBuffer buffer;
        const ID_type parentType;
    private:
        // queueItem* queue;
        short parentID;
        short queueLength;
        friend class AudioPipelineManager;
    };

}

#endif
