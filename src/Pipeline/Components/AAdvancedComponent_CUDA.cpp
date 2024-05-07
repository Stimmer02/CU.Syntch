#include "AAdvancedComponent_CUDA.h"

using namespace pipeline;

AAdvancedComponent_CUDA::AAdvancedComponent_CUDA(const audioFormatInfo* audioInfo, uint settingCount, const std::string* settingNames, short maxConnections, audioBufferQueue* boundBuffer): AComponent_CUDA(audioInfo, settingCount, settingNames, COMP_INVALID), maxConnections(maxConnections){
    connectionsCount = 0;
    connections =  new audioBufferQueue*[maxConnections];
    cudaMalloc(&d_connections, maxConnections * sizeof(advancedComponentConnection_CUDA));

    for (ushort i = 0; i < maxConnections; i++){
        connections[i] = nullptr;
    }

    includedIn = boundBuffer;
}

AAdvancedComponent_CUDA::~AAdvancedComponent_CUDA(){
    delete[] connections;
    cudaFree(d_connections);
}

void AAdvancedComponent_CUDA::connect(short index, audioBufferQueue* connectionBuffer){
    if (connections[index] == nullptr){
        connectionsCount++;
    }
    connections[index] = connectionBuffer;
    cudaMemcpy(&(d_connections[index].bufferL), &(connectionBuffer->buffer.d_bufferL), sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_connections[index].bufferR), &(connectionBuffer->buffer.d_bufferR), sizeof(float*), cudaMemcpyHostToDevice);
}

void AAdvancedComponent_CUDA::disconnect(ID_type IDType, short ID){
    for (ushort i = 0; i < maxConnections; i++){
        if (connections[i]->getParentID() == ID && connections[i]->parentType == IDType){
            connectionsCount--;
            connections[i] = nullptr;
        }
    }
}

void AAdvancedComponent_CUDA::disconnect(short index){
    if (connections[index] != nullptr){
        connectionsCount--;
        connections[index] = nullptr;
    }
}

short AAdvancedComponent_CUDA::getConnectionCount(){
    return connectionsCount;
}

void AAdvancedComponent_CUDA::getConnection(short index, ID_type& IDType, short& ID){
    if (connections[index] == nullptr){
        IDType = pipeline::INVALID;
        ID = -2;
    } else {
        IDType = connections[index]->parentType;
        ID = connections[index]->getParentID();
    }
}

audioBufferQueue* AAdvancedComponent_CUDA::getConnection(short index){
    return connections[index];
}

