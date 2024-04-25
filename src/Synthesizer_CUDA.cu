#include <cuda_runtime_api.h>
#include <cuda.h>

#include "Synthesizer/noteBuffer_CUDA.h"
#include "Synthesizer/settings_CUDA.h"

#define PI 3.1415926595
#define BASE_FREQUENCY 440.0

using namespace synthesizer;

__global__ void kernel_calculateFrequenciesSine(noteBuffer_CUDA* notes, settings_CUDA* settings){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < settings->keyCount){
        notes->frequency[i] = BASE_FREQUENCY * pow(2.0, (i + settings->pitch-69)/12.0);
        notes->multiplier[i] = PI*2 * notes->frequency[i] / settings->sampleRate;
    }
}

__global__ void kernel_calculateFrequenciesTriangle(noteBuffer_CUDA* notes, settings_CUDA* settings){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < settings->keyCount){
        notes->frequency[i] = BASE_FREQUENCY * pow(2.0, (i + settings->pitch-69)/12.0);
        notes->multiplier[i] = settings->sampleRate / notes->frequency[i] / 2;
    }
}

__global__ void kernel_calculateFrequenciesRest(noteBuffer_CUDA* notes, settings_CUDA* settings){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < settings->keyCount){
        notes->frequency[i] = BASE_FREQUENCY * pow(2.0, (i + settings->pitch-69)/12.0);
        notes->multiplier[i] = settings->sampleRate / notes->frequency[i];
    }
}

__global__ void kernel_calculateStereoFactor(noteBuffer_CUDA* notes, settings_CUDA* settings){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < settings->keyCount){
        float multiplierStep = (1-settings->stereoMix) / settings->keyCount;
        notes->stereoFactorL[i] = 1 - i * multiplierStep;
        notes->stereoFactorR[i] = settings->stereoMix + i * multiplierStep;
    }
}

__global__ void kernel_mixAudio(noteBuffer_CUDA* notes, settings_CUDA* settings, float* bufferL, float* bufferR){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < settings->sampleSize){
        bufferL = 0;
        bufferR = 0;
        for (uint j = 0; j < settings->keyCount; j++){
            uint noteBufferIndex = i + j * settings->sampleSize;
            bufferL[i] += notes->buffer[noteBufferIndex] * notes->stereoFactorL[j];
            bufferR[i] += notes->buffer[noteBufferIndex] * notes->stereoFactorR[j];
        }
    }
}
        