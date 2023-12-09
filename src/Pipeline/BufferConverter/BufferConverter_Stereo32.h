#ifndef IBUFFERCONVERTER_STEREO32_H
#define IBUFFERCONVERTER_STEREO32_H

#include "IBufferConverter.h"

class BufferConverter_Stereo32 : public IBufferConverter{
public:
    BufferConverter_Stereo32(){};
    ~BufferConverter_Stereo32()override{};
    void toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer) override;
};

#endif
