#ifndef IBUFFERCONVERTER_MONO8_H
#define IBUFFERCONVERTER_MONO8_H

#include "IBufferConverter.h"

class BufferConverter_Mono8 : public IBufferConverter{
public:
    ~BufferConverter_Mono8(){};
    void toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer) override;
};

#endif
