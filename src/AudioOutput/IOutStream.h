#ifndef _IOUTSTREAM_H
#define _IOUTSTREAM_H

#include <pulse/simple.h>
#include <pulse/error.h>
#include <string>

#include "audioBuffer.h"
#include "audioFormatInfo.h"

class IOutStream{
public:
    virtual ~IOutStream(){};
    virtual char init(const audioFormatInfo& info, const std::string& appName, const std::string& description) = 0;
    virtual char playBuffer(audioBuffer* buffer) = 0;
    virtual char flushBuffer() = 0;
    virtual char drainBuffer() = 0;
};

#endif
