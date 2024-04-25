#include <cuda_runtime_api.h>
#include <cuda.h>

#include "Generator_CUDA.h"


using namespace synthesizer;

__device__ float Generator_CUDA::kernel_soundSine(const float phaze, const float multiplier){
    return sin(phaze * multiplier);
}

__device__ float Generator_CUDA::kernel_soundSquare(const float phaze, const float multiplier){
    return (int((phaze) / multiplier) & 0x1)*2 - 1;
}

__device__ float Generator_CUDA::kernel_soundSawtooth(const float phaze, const float multiplier){
    float volume = phaze / multiplier;
    uint round = volume;
    volume -= round;
    return volume*2 - 1;
}

__device__ float Generator_CUDA::kernel_soundTriangle(const float phaze, const float multiplier){
    float volume = phaze / multiplier;
    uint round = volume;
    volume -= round;
    volume = (int(phaze / multiplier) & 0x1) == 0 ? volume : 1.0 - volume;
    return volume*2 - 1;
}

__device__ float Generator_CUDA::kernel_soundNoise1(const float phaze, const float multiplier){
    return sin(phaze * multiplier);
}

__global__ void Generator_CUDA::kernel_setGenerator(generator_type type){
    switch(type){
        case SINE:
            soundFunction = &Generator_CUDA::kernel_soundSine;
            break;
        case SQUARE:
            soundFunction = &Generator_CUDA::kernel_soundSquare;
            break;
        case SAWTOOTH:
            soundFunction = &Generator_CUDA::kernel_soundSawtooth;
            break;
        case TRIANGLE:
            soundFunction = &Generator_CUDA::kernel_soundTriangle;
            break;
        case NOISE1:
            soundFunction = &Generator_CUDA::kernel_soundNoise1;
            break;
    }
}

__global__ void Generator_CUDA::kernel_preprocessTimeDependentValues(noteBuffer_CUDA* noteBuffer, const uchar* keyState, const settings_CUDA* settings, const uint* releaseToAttackIndexMap, const float* dynamicsProfile){
    uint i = blockIdx.x * blockDim.x + threadIdx.x; // key index
    if (i < settings->keyCount){
        uint workArrIndex = i*settings->sampleSize;
        if (keyState[workArrIndex]){
            if (noteBuffer->lastKeyState[i] == 0 && noteBuffer->phaze[i] != 0){
                d_pressSamplessPassedWorkArr[workArrIndex] = releaseToAttackIndexMap[noteBuffer->releaseSamplesPassed[i]] * dynamicsProfile[noteBuffer->pressSamplessPassed[i]];
            } else {
                d_pressSamplessPassedWorkArr[workArrIndex] = noteBuffer->pressSamplessPassed[i] + 1;
            }
            d_pressSamplessPassedWorkArr[workArrIndex+1] = d_pressSamplessPassedWorkArr[workArrIndex] + 1;
            d_phazeWorkArr[workArrIndex] = noteBuffer->phaze[i] + 1;
            d_releaseSamplesPassedWorkArr[workArrIndex+1] = 0;
            d_velocityWorkArr[workArrIndex] = keyState[workArrIndex]/127.0;
        } else if (noteBuffer->releaseSamplesPassed[i] < settings->release.duration){
            d_phazeWorkArr[workArrIndex] = noteBuffer->phaze[i] + 1;
            d_releaseSamplesPassedWorkArr[workArrIndex] = noteBuffer->releaseSamplesPassed[i];
            d_releaseSamplesPassedWorkArr[workArrIndex+1] = d_releaseSamplesPassedWorkArr[workArrIndex] + 1;
            d_velocityWorkArr[workArrIndex] = noteBuffer->velocity[i];
            if (keyState[workArrIndex+1]){
                d_pressSamplessPassedWorkArr[workArrIndex+1] = releaseToAttackIndexMap[d_releaseSamplesPassedWorkArr[workArrIndex]] * dynamicsProfile[d_pressSamplessPassedWorkArr[workArrIndex]]; 
            } else {
                d_pressSamplessPassedWorkArr[workArrIndex+1] = d_pressSamplessPassedWorkArr[workArrIndex];
            }
        } else {
            d_phazeWorkArr[workArrIndex] = 0;
            d_pressSamplessPassedWorkArr[workArrIndex+1] = 0;
            noteBuffer->buffer[workArrIndex] = 0;
        }

        for (uint j = 1; j < settings->sampleSize - 1; j++){
            workArrIndex = j + i*settings->sampleSize;
            if (keyState[workArrIndex]){
                d_phazeWorkArr[workArrIndex] = d_phazeWorkArr[workArrIndex-1] + 1;
                d_pressSamplessPassedWorkArr[workArrIndex+1] = d_pressSamplessPassedWorkArr[workArrIndex] + 1;
                d_releaseSamplesPassedWorkArr[workArrIndex+1] = 0;
                d_velocityWorkArr[workArrIndex] = keyState[workArrIndex]/127.0;
            } else if (noteBuffer->releaseSamplesPassed[i] < settings->release.duration){
                d_phazeWorkArr[workArrIndex] = d_phazeWorkArr[workArrIndex-1] + 1;
                if (keyState[workArrIndex+1]){
                    d_pressSamplessPassedWorkArr[workArrIndex+1] = releaseToAttackIndexMap[d_releaseSamplesPassedWorkArr[workArrIndex]] * dynamicsProfile[d_pressSamplessPassedWorkArr[workArrIndex]]; 
                } else {
                    d_pressSamplessPassedWorkArr[workArrIndex+1] = d_pressSamplessPassedWorkArr[workArrIndex];
                }
                d_releaseSamplesPassedWorkArr[workArrIndex+1] = d_releaseSamplesPassedWorkArr[workArrIndex] + 1;
                d_velocityWorkArr[workArrIndex] = d_velocityWorkArr[workArrIndex-1];
            } else {
                d_phazeWorkArr[workArrIndex] = 0;
                noteBuffer->buffer[workArrIndex] = 0;
                d_pressSamplessPassedWorkArr[workArrIndex+1] = 0;
            }
        }
        workArrIndex = settings->sampleSize-1 + i*settings->sampleSize;
        if (keyState[workArrIndex]){
            d_phazeWorkArr[workArrIndex] = d_phazeWorkArr[workArrIndex-1] + 1;
            noteBuffer->pressSamplessPassed[i] = d_pressSamplessPassedWorkArr[workArrIndex] + 1;
            noteBuffer->releaseSamplesPassed[i] = 0;
            d_velocityWorkArr[workArrIndex] = keyState[workArrIndex]/127.0;
        } else if (noteBuffer->releaseSamplesPassed[i] < settings->release.duration){
            d_phazeWorkArr[workArrIndex] = d_phazeWorkArr[workArrIndex-1] + 1;
            noteBuffer->pressSamplessPassed[i] = d_pressSamplessPassedWorkArr[workArrIndex];
            noteBuffer->releaseSamplesPassed[i] = d_releaseSamplesPassedWorkArr[workArrIndex] + 1;
            d_velocityWorkArr[workArrIndex] = d_velocityWorkArr[workArrIndex-1];
        } else {
            d_phazeWorkArr[workArrIndex] = 0;
            noteBuffer->buffer[workArrIndex] = 0;
            noteBuffer->releaseSamplesPassed[i] = 0;
            noteBuffer->pressSamplessPassed[i] = 0;
        }

        noteBuffer->phaze[i] = d_phazeWorkArr[workArrIndex];
        noteBuffer->velocity[i] = d_velocityWorkArr[workArrIndex];
        noteBuffer->lastKeyState[i] = keyState[workArrIndex];
    }
}

__global__  void Generator_CUDA::kernel_generate(noteBuffer_CUDA* noteBuffer, const uchar* keyState, const settings_CUDA* settings, const float* dynamicsProfile, const float* releaseProfile){
    uint i = blockIdx.x * blockDim.x + threadIdx.x; // key index
    uint j = blockIdx.y * blockDim.y + threadIdx.y; // sample index
    if (i < settings->keyCount && keyState[i] && j < settings->sampleSize){
        uint workArrIndex = j + i*settings->sampleSize;
        float dynamicsMultiplier = settings->dynamicsDuration > d_pressSamplessPassedWorkArr[workArrIndex] ? dynamicsProfile[d_pressSamplessPassedWorkArr[workArrIndex]] : settings->fadeTo;

        if (d_releaseSamplesPassedWorkArr[workArrIndex] < settings->release.duration){
            dynamicsMultiplier *= releaseProfile[d_releaseSamplesPassedWorkArr[workArrIndex]];
        }

        noteBuffer->buffer[workArrIndex] = (*this.*soundFunction)(d_phazeWorkArr[workArrIndex], noteBuffer->multiplier[i]) * settings->volume * d_velocityWorkArr[workArrIndex] * dynamicsMultiplier;
    }
}
