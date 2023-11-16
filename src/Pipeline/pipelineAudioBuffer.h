#ifndef _PIPELINEAUDIOBUFFER_H
#define _PIPELINEAUDIOBUFFER_H

typedef unsigned int uint;

struct pipelineAudioBuffer {
    int* buffer;
    const uint size;

    pipelineAudioBuffer(const uint& sampleSize) : size(sampleSize){
        buffer = new int[sampleSize];
    }
    ~pipelineAudioBuffer(){
        delete[] buffer;
    }
};

#endif
