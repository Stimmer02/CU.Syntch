#ifndef KEYBOARDDOUBLEBUFFER_EMPTY_H
#define KEYBOARDDOUBLEBUFFER_EMPTY_H

#include <chrono>
#include "IKeyboardDoubleBuffer.h"

typedef unsigned int uint;
typedef unsigned short int ushort;
typedef unsigned char uchar;

class KeyboardDoubleBuffer_Empty: public IKeyboardDoubleBuffer{

public:
    KeyboardDoubleBuffer_Empty(const uint& sampleSize, const ushort& keyCount);
    ~KeyboardDoubleBuffer_Empty() override;
    uchar** getInactiveBuffer() override;
    uchar** getActiveBuffer() override;
    void swapActiveBuffer() override;
    void clearInactiveBuffer() override;
    long getActivationTimestamp() override;

    ushort getKeyCount() override;
    uint getSampleSize() override;

private:
    const ushort keyCount;
    const uint sampleSize;

    long activationTimestamp;
    uchar** buffer;//first dimension is keyCount second sampleSize
};

#endif
