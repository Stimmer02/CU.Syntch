#ifndef AADVANCEDCOMPONENT_CUDA_H
#define AADVANCEDCOMPONENT_CUDA_H

#include "AComponent_CUDA.h"
#include "../audioBufferQueue.h"
#include "advancedComponentConnection_CUDA.h"

namespace pipeline{
    enum advanced_component_type{
        ACOMP_INVALID,
        ACOMP_SUM2,
        ACOMP_SUM7,
        ACOMP_COPY,
    };

    class AAdvancedComponent_CUDA: public AComponent_CUDA{
    public:
        AAdvancedComponent_CUDA(const audioFormatInfo* audioInfo, uint settingCount, const std::string* settingNames, short maxConnections, audioBufferQueue* boundBuffer);
        ~AAdvancedComponent_CUDA();

        virtual bool allNeededConnections() = 0;

        void connect(short index, audioBufferQueue* connectionBuffer);
        void disconnect(ID_type IDType, short ID);
        void disconnect(short index);
        void getConnection(short index, ID_type& IDType, short& ID);
        audioBufferQueue* getConnection(short index);
        short getConnectionCount();

        const short maxConnections;
    protected:
        short connectionsCount;
        audioBufferQueue** connections;
        advancedComponentConnection_CUDA* d_connections;
    };
}

#endif
