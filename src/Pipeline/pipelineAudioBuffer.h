#ifndef _PIPELINEAUDIOBUFFER_H
#define _PIPELINEAUDIOBUFFER_H

typedef unsigned int uint;

struct pipelineAudioBuffer {
    double* bufferL;
    double* bufferR;
    const uint size;

    pipelineAudioBuffer(const uint& sampleSize) : size(sampleSize){
        bufferL = new double[sampleSize];
        bufferR = new double[sampleSize];
    }
    ~pipelineAudioBuffer(){
        delete[] bufferL;
        delete[] bufferR;
    }
};

#endif
