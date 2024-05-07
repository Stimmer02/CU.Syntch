#ifndef IBUFFERCONVERTER_MONO16_CUDA_H
#define IBUFFERCONVERTER_MONO16_CUDA_H

#include "IBufferConverter_CUDA.h"

class BufferConverter_Mono16_CUDA : public IBufferConverter_CUDA{
public:
    BufferConverter_Mono16_CUDA(const uint& sampleSize);
    ~BufferConverter_Mono16_CUDA();
    void toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer) override;

private:
    uint8_t* d_buffer;
    const uint sampleSize;
};

#endif
