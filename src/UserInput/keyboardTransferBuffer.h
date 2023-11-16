#ifndef _KEYBOARDTRANSFERBUFFER_H
#define _KEYBOARDTRANSFERBUFFER_H

#include "KeyboardDoubleBuffer.h"

struct keyboardTransferBuffer{
    keyboardTransferBuffer(const uint& sampleSize, const ushort& keyCount);
    ~keyboardTransferBuffer();
    void convertBuffer(KeyboardDoubleBuffer* keyboardBuffer);

    uchar** buffer;
    uchar* lastState;
    const uint sampleSize;
    const ushort keyCount;
};

#endif
