#ifndef AADVANCEDCOMPONENT_H
#define AADVANCEDCOMPONENT_H

#include "AComponent.h"
#include "../audioBufferQueue.h"

namespace pipeline{
    enum advanced_component_type{
        ACOMP_INVALID,
        ACOMP_SUM2,
        ACOMP_COPY,
    };

    class AAdvancedComponent: public AComponent{
    public:
        AAdvancedComponent(const audioFormatInfo* audioInfo, uint settingCount, const std::string* settingNames, short maxConnections, audioBufferQueue* boundBuffer);
        ~AAdvancedComponent();

        virtual bool allNeededConnections() = 0;

        void connect(short index, audioBufferQueue* connectionBuffer);
        void disconnect(ID_type IDType, short ID);
        void disconnect(short index);
        void getConnection(short index, ID_type& IDType, short& ID);
        audioBufferQueue* getConnection(short index);
        short getConnectionCount();

        const ushort maxConnections;
    protected:
        short connectionsCount;
        audioBufferQueue** connections;
    };
}

#endif
