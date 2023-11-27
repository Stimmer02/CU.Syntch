#ifndef _IBUFFERCONVERTER_STEREO24_H
#define _IBUFFERCONVERTER_STEREO24_H

#include "IBufferConverter.h"

class BufferConverter_Stereo24 : public IBufferConverter{
public:
    BufferConverter_Stereo24(){};
    ~BufferConverter_Stereo24()override{};
    void toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer) override;
};

#endif
