#ifndef _IBUFFERCONVERTER_STEREO8_H
#define _IBUFFERCONVERTER_STEREO8_H

#include "IBufferConverter.h"

class BufferConverter_Stereo8 : public IBufferConverter{
public:
    ~BufferConverter_Stereo8(){};
    void toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer) override;
};

#endif
