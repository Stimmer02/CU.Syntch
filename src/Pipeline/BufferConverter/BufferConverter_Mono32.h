#ifndef IBUFFERCONVERTER_MONO32_H
#define IBUFFERCONVERTER_MONO32_H

#include "IBufferConverter.h"

class BufferConverter_Mono32 : public IBufferConverter{
public:
    BufferConverter_Mono32(){};
    ~BufferConverter_Mono32()override{};
    void toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer) override;
};

#endif
