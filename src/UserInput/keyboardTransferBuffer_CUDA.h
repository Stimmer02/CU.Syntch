#ifndef KEYBOARDTRANSFERBUFFER_CUDA_H
#define KEYBOARDTRANSFERBUFFER_CUDA_H

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "IKeyboardDoubleBuffer.h"

struct keyboardTransferBuffer_CUDA{
    keyboardTransferBuffer_CUDA(const uint& sampleSize, const ushort& keyCount);
    ~keyboardTransferBuffer_CUDA();
    void convertBuffer(IKeyboardDoubleBuffer* keyboardBuffer);
    void convertBuffer(uchar* buff[127]);

    uchar* d_buffer; //2D array: [keyCount][sampleSize]
    uchar* d_input; //2D array: [keyCount][sampleSize]
    uchar* d_lastState;
    const uint sampleSize;
    const ushort keyCount;
};

#endif
