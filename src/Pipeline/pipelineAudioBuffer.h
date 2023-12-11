#ifndef PIPELINEAUDIOBUFFER_H
#define PIPELINEAUDIOBUFFER_H

#include <cstdlib>

typedef unsigned int uint;

struct pipelineAudioBuffer {
    float* bufferL;
    float* bufferR;
    const uint size;

    pipelineAudioBuffer(const uint& sampleSize) : size(sampleSize){
        bufferL = static_cast<float*>(std::aligned_alloc(32, sampleSize * sizeof(float)));
        bufferR = static_cast<float*>(std::aligned_alloc(32, sampleSize * sizeof(float)));
    }
    ~pipelineAudioBuffer(){
        free(bufferL);
        free(bufferR);
    }
};

#endif
