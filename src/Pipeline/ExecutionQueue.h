#ifndef EXECUTIONQUEUE_H
#define EXECUTIONQUEUE_H

#include "AudioBufferQueue.h"
#include <vector>

namespace pipeline{
    class ExecutionQueue{
    public:
        char build(std::vector<AudioBufferQueue*>& componentQueues, AudioBufferQueue*& outputQueue);
        const std::vector<AudioBufferQueue*>& getQueue();
        const std::vector<short>& getConnectedSynthIDs();
        bool valid();

    private:
        std::vector<AudioBufferQueue*> operations;
        std::vector<short> connectedSynthIDs;
    };
}

#endif
