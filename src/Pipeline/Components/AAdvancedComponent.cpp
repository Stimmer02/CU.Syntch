#include "AAdvancedComponent.h"

using namespace pipeline;

AAdvancedComponent::AAdvancedComponent(const audioFormatInfo* audioInfo, uint settingCount, const std::string* settingNames, short maxConnections, audioBufferQueue* boundBuffer): AComponent(audioInfo, settingCount, settingNames, COMP_INVALID), maxConnections(maxConnections){
    connectionsCount = 0;
    connections =  new audioBufferQueue*[maxConnections];

    for (ushort i = 0; i < maxConnections; i++){
        connections[i] = nullptr;
    }

    includedIn = boundBuffer;
}

AAdvancedComponent::~AAdvancedComponent(){
    delete[] connections;
}

void AAdvancedComponent::connect(short index, audioBufferQueue* connectionBuffer){
    if (connections[index] == nullptr){
        connectionsCount++;
    }
    connections[index] = connectionBuffer;
}

void AAdvancedComponent::disconnect(ID_type IDType, short ID){
    for (ushort i = 0; i < maxConnections; i++){
        if (connections[i]->getParentID() == ID && connections[i]->parentType == IDType){
            connectionsCount--;
            connections[i] = nullptr;
        }
    }
}

void AAdvancedComponent::disconnect(short index){
    if (connections[index] != nullptr){
        connectionsCount--;
        connections[index] = nullptr;
    }
}

short AAdvancedComponent::getConnectionCount(){
    return connectionsCount;
}

void AAdvancedComponent::getConnection(short index, ID_type& IDType, short& ID){
    if (connections[index] == nullptr){
        IDType = pipeline::INVALID;
        ID = -2;
    } else {
        IDType = connections[index]->parentType;
        ID = connections[index]->getParentID();
    }
}

audioBufferQueue* AAdvancedComponent::getConnection(short index){
    return connections[index];
}

