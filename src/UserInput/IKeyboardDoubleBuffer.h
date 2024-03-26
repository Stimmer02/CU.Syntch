#ifndef IKEYBOARDDOUBLEBUFFER_H
#define IKEYBOARDDOUBLEBUFFER_H


typedef unsigned int uint;
typedef unsigned short int ushort;
typedef unsigned char uchar;

class IKeyboardDoubleBuffer{

public:
    IKeyboardDoubleBuffer(){};
    virtual ~IKeyboardDoubleBuffer() = default;
    virtual uchar** getInactiveBuffer() = 0;
    virtual uchar** getActiveBuffer() = 0;
    virtual void swapActiveBuffer() = 0;
    virtual void clearInactiveBuffer() = 0;
    virtual long getActivationTimestamp() = 0;

    virtual ushort getKeyCount() = 0;
    virtual uint getSampleSize() = 0;
};

#endif
