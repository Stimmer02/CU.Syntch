#ifndef NOTEBUFFER_CUDA_H
#define NOTEBUFFER_CUDA_H

typedef unsigned int uint;
typedef unsigned char uchar;

namespace synthesizer{
    struct noteBuffer_CUDA{
        float* buffer; //2D array: [keyCount][sampleSize]
        float* velocity;
        char* lastKeyState;

        uint* phaze;
        uint* pressSamplessPassed;
        uint* releaseSamplesPassed;

        float* stereoFactorL;
        float* stereoFactorR;

        float* frequency;
        float* multiplier;
    };
}

#endif
