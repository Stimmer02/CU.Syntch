#ifndef IBUFFERCONVERTER_CUDA_H
#define IBUFFERCONVERTER_CUDA_H

#include "../../AudioOutput/audioBuffer.h"
#include "../pipelineAudioBuffer_CUDA.h"

#include <cuda_runtime.h>
#include <cuda.h>

#define CUDA_BUFFER_CONVERTER_BLOCK_SIZE 256

class IBufferConverter_CUDA{
public:
    virtual ~IBufferConverter_CUDA(){};
    virtual void toPCM(pipelineAudioBuffer_CUDA* pipelineBuffer, audioBuffer* pcmBuffer) = 0;
};

#endif
