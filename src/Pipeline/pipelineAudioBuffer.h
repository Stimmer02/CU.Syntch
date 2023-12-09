#ifndef PIPELINEAUDIOBUFFER_H
#define PIPELINEAUDIOBUFFER_H

typedef unsigned int uint;

struct pipelineAudioBuffer {
    float* bufferL;
    float* bufferR;
    const uint size;

    pipelineAudioBuffer(const uint& sampleSize) : size(sampleSize){
        bufferL = new float[sampleSize];
        bufferR = new float[sampleSize];
    }
    ~pipelineAudioBuffer(){
        delete[] bufferL;
        delete[] bufferR;
    }
};

#endif
