#ifndef EXECUTIONQUEUE_H
#define EXECUTIONQUEUE_H

#include "audioBufferQueue.h"
#include "UniqueShiftBuffer.h"
#include "Components/AAdvancedComponent.h"
#include <vector>

namespace pipeline{
    class ExecutionQueue{
    public:
        void build(std::vector<audioBufferQueue*>& componentQueues, audioBufferQueue*& outputQueue, IDManager<AComponent, short>& components);
        const std::vector<audioBufferQueue*>& getQueue();
        const std::vector<short>& getConnectedSynthIDs();
        char error();

        const std::vector<short>& getInvalidAdvCompIDs();

    private:
        std::vector<audioBufferQueue*> operations;
        std::vector<short> connectedSynthIDs;

        std::vector<short> invalidAdvancedComponents;
    };
}

#endif
