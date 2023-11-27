#ifndef _IBUFFERCONVERTER_STEREO16_H
#define _IBUFFERCONVERTER_STEREO16_H

#include "IBufferConverter.h"

class BufferConverter_Stereo16 : public IBufferConverter{
public:
    ~BufferConverter_Stereo16(){};
    void toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer) override;
};

#endif
