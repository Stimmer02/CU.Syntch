#ifndef AUDIOBUFFER_H
#define AUDIOBUFFER_H

#include <inttypes.h>
#include <sys/types.h>

struct audioBuffer {
    uint8_t* buff;
    uint32_t count;
    const uint32_t size;

    audioBuffer(const uint32_t& buffSize);
    ~audioBuffer();
};

#endif
