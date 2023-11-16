#ifndef _AUDIOBUFFER_H
#define _AUDIOBUFFER_H

#include <inttypes.h>
#include <sys/types.h>

struct audioBuffer {
    uint8_t* buff;
    ssize_t count;
    const uint16_t size;

    audioBuffer(const uint16_t& buffSize);
    ~audioBuffer();
};

#endif
