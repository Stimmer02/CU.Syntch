#ifndef _IBUFFERCONVERTER_MONO16_H
#define _IBUFFERCONVERTER_MONO16_H

#include "IBufferConverter.h"

class BufferConverter_Mono16 : public IBufferConverter{
public:
    ~BufferConverter_Mono16(){};
    void toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer) override;
};

#endif
