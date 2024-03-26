#ifndef KEYBOARDTRANSFERBUFFER_H
#define KEYBOARDTRANSFERBUFFER_H

#include "IKeyboardDoubleBuffer.h"

struct keyboardTransferBuffer{
    keyboardTransferBuffer(const uint& sampleSize, const ushort& keyCount);
    ~keyboardTransferBuffer();
    void convertBuffer(IKeyboardDoubleBuffer* keyboardBuffer);
    void convertBuffer(uchar* buff[127]);

    uchar** buffer;
    uchar* lastState;
    const uint sampleSize;
    const ushort keyCount;
};

#endif
