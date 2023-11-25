#ifndef _KEYBOARDDOUBLEBUFFER_H
#define _KEYBOARDDOUBLEBUFFER_H

#include <chrono>

typedef unsigned int uint;
typedef unsigned short int ushort;
typedef unsigned char uchar;

class KeyboardDoubleBuffer {

public:
    KeyboardDoubleBuffer(const uint& sampleSize, const ushort& keyCount);
    ~KeyboardDoubleBuffer();
    uchar** getInactiveBuffer();
    uchar** getActiveBuffer();
    void swapActiveBuffer();
    void clearInactiveBuffer();
    long getActivationTimestamp();

private:
    const ushort keyCount;
    const uint sampleSize;

    bool activeBuffer;
    long activationTimestamp[2];
    uchar** buffer[2];//first dimension is keyCount second sampleSize
};

#endif
