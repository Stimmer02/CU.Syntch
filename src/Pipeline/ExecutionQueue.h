#ifndef EXECUTIONQUEUE_H
#define EXECUTIONQUEUE_H

#include "audioBufferQueue.h"
#include <vector>

namespace pipeline{
    class ExecutionQueue{
    public:
        char build(std::vector<audioBufferQueue*>& componentQueues, audioBufferQueue*& outputQueue);
        const std::vector<audioBufferQueue*>& getQueue();
        const std::vector<short>& getConnectedSynthIDs();
        bool valid();

    private:
        std::vector<audioBufferQueue*> operations;
        std::vector<short> connectedSynthIDs;
    };
}

#endif
