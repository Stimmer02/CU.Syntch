#ifndef _IBUFFERCONVERTER_H
#define _IBUFFERCONVERTER_H

#include "../../AudioOutput/audioBuffer.h"
#include "../pipelineAudioBuffer.h"

class IBufferConverter{
public:
    virtual ~IBufferConverter(){};
    virtual void toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer) = 0;
};

#endif
