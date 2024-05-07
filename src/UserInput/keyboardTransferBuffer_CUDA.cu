#include "keyboardTransferBuffer_CUDA.h"

__global__ void kernel_convertBuffer(const uint sampleSize, const ushort keyCount, uchar* input, uchar* buffer, uchar* lastState){
    const uint i = blockIdx.x * blockDim.x + threadIdx.x; // keyCount
    if (i < keyCount){
        uchar lastStateTemp = lastState[i];
        for (uint j = 0; j < sampleSize; j++){
            uint index = i * sampleSize + j;
            if (input[index] == 255){
                lastStateTemp = 0;
            } else if (input[index] > 0){
                lastStateTemp = input[index];
            }
            buffer[index] = lastStateTemp;
        }
        lastState[i] = lastStateTemp;
    }    
}


keyboardTransferBuffer_CUDA::keyboardTransferBuffer_CUDA(const uint& sampleSize, const ushort& keyCount) : sampleSize(sampleSize), keyCount(keyCount){
    cudaMalloc((void**)(&d_buffer), keyCount * sampleSize * sizeof(uchar));
    cudaMalloc((void**)(&d_input), keyCount * sampleSize * sizeof(uchar));
    cudaMalloc((void**)(&d_lastState), keyCount * sizeof(uchar));
}

keyboardTransferBuffer_CUDA::~keyboardTransferBuffer_CUDA(){
    cudaFree(d_buffer);
    cudaFree(d_input);
    cudaFree(d_lastState);
}

void keyboardTransferBuffer_CUDA::convertBuffer(IKeyboardDoubleBuffer* keyboardBuffer){
    convertBuffer(keyboardBuffer->getInactiveBuffer());
}

void keyboardTransferBuffer_CUDA::convertBuffer(uchar* buff[127]){
    static const uint blockSize = 128;

    for (uint i = 0; i < keyCount; i++){
        cudaMemcpy(d_input + i * sampleSize, buff[i], sampleSize * sizeof(uchar), cudaMemcpyHostToDevice);
    }

    uint blockCount = (keyCount + blockSize - 1) / blockSize;
    kernel_convertBuffer<<<blockCount, blockSize>>>(sampleSize, keyCount, d_input, d_buffer, d_lastState);
}



