#ifndef IBUFFERCONVERTER_MONO8_CUDA_H
#define IBUFFERCONVERTER_MONO8_CUDA_H

#include "IBufferConverter_CUDA.h"

class BufferConverter_Mono8_CUDA : public IBufferConverter_CUDA{
public:
    BufferConverter_Mono8_CUDA(const uint& sampleSize);
    ~BufferConverter_Mono8_CUDA();
    void toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer) override;
    
private:
    uint8_t* d_buffer;
    const uint sampleSize;
};

#endif
