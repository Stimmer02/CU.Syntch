#ifndef IBUFFERCONVERTER_H
#define IBUFFERCONVERTER_H

#include "../../AudioOutput/audioBuffer.h"
#include "../pipelineAudioBuffer.h"
#include <immintrin.h>

class IBufferConverter{
public:
    virtual ~IBufferConverter(){};
    virtual void toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer) = 0;
};

#endif
