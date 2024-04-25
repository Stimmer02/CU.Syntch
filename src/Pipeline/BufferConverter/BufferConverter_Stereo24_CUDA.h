#ifndef IBUFFERCONVERTER_STEREO24_CUDA_H
#define IBUFFERCONVERTER_STEREO24_CUDA_H

#include "IBufferConverter_CUDA.h"

class BufferConverter_Stereo24_CUDA : public IBufferConverter_CUDA{
public:
    BufferConverter_Stereo24_CUDA(const uint& sampleSize);
    ~BufferConverter_Stereo24_CUDA();
    void toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer) override;

private:
    __global__ void toPCM_kernel(const float* bufferL, const float* bufferR);
    uint8_t* d_buffer;
    const uint sampleSize;
};

#endif
