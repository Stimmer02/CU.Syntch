#ifndef IBUFFERCONVERTER_MONO24_H
#define IBUFFERCONVERTER_MONO24_H

#include "IBufferConverter.h"

class BufferConverter_Mono24 : public IBufferConverter{
public:
    BufferConverter_Mono24(){};
    ~BufferConverter_Mono24()override{};
    void toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer) override;
};

#endif
