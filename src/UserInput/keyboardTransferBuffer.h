#ifndef _KEYBOARDTRANSFERBUFFER_H
#define _KEYBOARDTRANSFERBUFFER_H

#include "KeyboardDoubleBuffer.h"

struct keyboardTransferBuffer{
    keyboardTransferBuffer(const uint& sampleSize, const ushort& keyCount);
    ~keyboardTransferBuffer();
    void convertBuffer(KeyboardDoubleBuffer* keyboardBuffer);
    void convertBuffer(uchar* buff[127]);

    uchar** buffer;
    uchar* lastState;
    const uint sampleSize;
    const ushort keyCount;
};

#endif
