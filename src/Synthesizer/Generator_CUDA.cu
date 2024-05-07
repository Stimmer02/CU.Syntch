#include "Generator_CUDA.h"
#include "stdio.h"

using namespace synthesizer;

__device__ float kernel_soundSine(const float phaze, const float multiplier){
    return sin(phaze * multiplier);
}

__device__ float kernel_soundSquare(const float phaze, const float multiplier){
    return (int((phaze) / multiplier) & 0x1)*2 - 1;
}

__device__ float kernel_soundSawtooth(const float phaze, const float multiplier){
    float volume = phaze / multiplier;
    uint round = volume;
    volume -= round;
    return volume*2 - 1;
}

__device__ float kernel_soundTriangle(const float phaze, const float multiplier){
    float volume = phaze / multiplier;
    uint round = volume;
    volume -= round;
    volume = (int(phaze / multiplier) & 0x1) == 0 ? volume : 1.0 - volume;
    return volume*2 - 1;
}

__device__ float kernel_soundNoise1(const float phaze, const float multiplier){
    return sin(phaze * multiplier);
}

// __global__ void kernel_setGenerator(const generator_type type, Generator_CUDA::soundFunctionPointer* soundFunction){
//     switch(type){
//         case SINE:
//             *soundFunction = &kernel_soundSine;
//             break;
//         case SQUARE:
//             *soundFunction = &kernel_soundSquare;
//             break;
//         case SAWTOOTH:
//             *soundFunction = &kernel_soundSawtooth;
//             break;
//         case TRIANGLE:
//             *soundFunction = &kernel_soundTriangle;
//             break;
//         case NOISE1:
//             *soundFunction = &kernel_soundNoise1;
//             break;
//     }
// }

__global__ void kernel_initFunctionArray(Generator_CUDA::soundFunctionPointer* functionArray){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        functionArray[SINE] = &kernel_soundSine;
        functionArray[SQUARE] = &kernel_soundSquare;
        functionArray[SAWTOOTH] = &kernel_soundSawtooth;
        functionArray[TRIANGLE] = &kernel_soundTriangle;
        functionArray[NOISE1] = &kernel_soundNoise1;
    }
}

__global__ void kernel_preprocessTimeDependentValues(noteBuffer_CUDA* noteBuffer, const uchar* keyState, const settings_CUDA* settings, const uint* releaseToAttackIndexMap, const float* dynamicsProfile, uint* phazeWorkArr, uint* pressSamplessPassedWorkArr, uint* releaseSamplesPassedWorkArr, float* velocityWorkArr){
    uint i = blockIdx.x * blockDim.x + threadIdx.x; // key index
    if (i < settings->keyCount){
        uint workArrIndex = i*settings->sampleSize;
        if (keyState[workArrIndex]){
            if (noteBuffer->lastKeyState[i] == 0 && noteBuffer->phaze[i] != 0){
                pressSamplessPassedWorkArr[workArrIndex] = releaseToAttackIndexMap[noteBuffer->releaseSamplesPassed[i]] * dynamicsProfile[noteBuffer->pressSamplessPassed[i]];
            } else {
                pressSamplessPassedWorkArr[workArrIndex] = noteBuffer->pressSamplessPassed[i] + 1;
            }
            pressSamplessPassedWorkArr[workArrIndex+1] = pressSamplessPassedWorkArr[workArrIndex] + 1;
            phazeWorkArr[workArrIndex] = noteBuffer->phaze[i] + 1;
            releaseSamplesPassedWorkArr[workArrIndex+1] = 0;
            releaseSamplesPassedWorkArr[workArrIndex] = noteBuffer->releaseSamplesPassed[i];
            velocityWorkArr[workArrIndex] = keyState[workArrIndex]/127.0;
        } else if (noteBuffer->releaseSamplesPassed[i] < settings->release.duration){
            phazeWorkArr[workArrIndex] = noteBuffer->phaze[i] + 1;
            releaseSamplesPassedWorkArr[workArrIndex] = noteBuffer->releaseSamplesPassed[i];
            releaseSamplesPassedWorkArr[workArrIndex+1] = releaseSamplesPassedWorkArr[workArrIndex] + 1;
            velocityWorkArr[workArrIndex] = noteBuffer->velocity[i];
            if (keyState[workArrIndex+1]){
                pressSamplessPassedWorkArr[workArrIndex+1] = releaseToAttackIndexMap[releaseSamplesPassedWorkArr[workArrIndex]] * dynamicsProfile[pressSamplessPassedWorkArr[workArrIndex]]; 
            } else {
                pressSamplessPassedWorkArr[workArrIndex+1] = pressSamplessPassedWorkArr[workArrIndex];
            }
        } else {
            releaseSamplesPassedWorkArr[workArrIndex] = noteBuffer->releaseSamplesPassed[i];
            releaseSamplesPassedWorkArr[workArrIndex+1] = noteBuffer->releaseSamplesPassed[i];
            phazeWorkArr[workArrIndex] = 0;
            pressSamplessPassedWorkArr[workArrIndex+1] = 0;
            noteBuffer->buffer[workArrIndex] = 0;
        }

        for (uint j = 1; j < settings->sampleSize - 1; j++){
            workArrIndex = j + i*settings->sampleSize;
            if (keyState[workArrIndex]){
                phazeWorkArr[workArrIndex] = phazeWorkArr[workArrIndex-1] + 1;
                pressSamplessPassedWorkArr[workArrIndex+1] = pressSamplessPassedWorkArr[workArrIndex] + 1;
                releaseSamplesPassedWorkArr[workArrIndex+1] = 0;
                velocityWorkArr[workArrIndex] = keyState[workArrIndex]/127.0;
                // printf("0 - %d\n", workArrIndex);
            } else if (releaseSamplesPassedWorkArr[workArrIndex] < settings->release.duration){
                phazeWorkArr[workArrIndex] = phazeWorkArr[workArrIndex-1] + 1;
                if (keyState[workArrIndex+1]){
                    pressSamplessPassedWorkArr[workArrIndex+1] = releaseToAttackIndexMap[releaseSamplesPassedWorkArr[workArrIndex]] * dynamicsProfile[pressSamplessPassedWorkArr[workArrIndex]]; 
                } else {
                    pressSamplessPassedWorkArr[workArrIndex+1] = pressSamplessPassedWorkArr[workArrIndex];
                }
                releaseSamplesPassedWorkArr[workArrIndex+1] = releaseSamplesPassedWorkArr[workArrIndex] + 1;
                velocityWorkArr[workArrIndex] = velocityWorkArr[workArrIndex-1];
                // printf("1 - %d; %d < %d\n", workArrIndex, releaseSamplesPassedWorkArr[workArrIndex], settings->release.duration);
            } else {
                releaseSamplesPassedWorkArr[workArrIndex+1] = releaseSamplesPassedWorkArr[workArrIndex];
                phazeWorkArr[workArrIndex] = 0;
                noteBuffer->buffer[workArrIndex] = 0;
                pressSamplessPassedWorkArr[workArrIndex+1] = 0;
                // if (i == 0) printf("2 - %d\n", workArrIndex);
            }
        }
        workArrIndex = settings->sampleSize-1 + i*settings->sampleSize;
        if (keyState[workArrIndex]){
            phazeWorkArr[workArrIndex] = phazeWorkArr[workArrIndex-1] + 1;
            noteBuffer->pressSamplessPassed[i] = pressSamplessPassedWorkArr[workArrIndex] + 1;
            noteBuffer->releaseSamplesPassed[i] = 0;
            velocityWorkArr[workArrIndex] = keyState[workArrIndex]/127.0;
        } else if (releaseSamplesPassedWorkArr[workArrIndex] < settings->release.duration){
            phazeWorkArr[workArrIndex] = phazeWorkArr[workArrIndex-1] + 1;
            noteBuffer->pressSamplessPassed[i] = pressSamplessPassedWorkArr[workArrIndex];
            noteBuffer->releaseSamplesPassed[i] = releaseSamplesPassedWorkArr[workArrIndex] + 1;
            velocityWorkArr[workArrIndex] = velocityWorkArr[workArrIndex-1];
        } else {
            phazeWorkArr[workArrIndex] = 0;
            noteBuffer->buffer[workArrIndex] = 0;
            noteBuffer->releaseSamplesPassed[i] = releaseSamplesPassedWorkArr[workArrIndex];
            noteBuffer->pressSamplessPassed[i] = 0;
        }

        noteBuffer->phaze[i] = phazeWorkArr[workArrIndex];
        noteBuffer->velocity[i] = velocityWorkArr[workArrIndex];
        noteBuffer->lastKeyState[i] = keyState[workArrIndex];
    }
}

__global__  void kernel_generate(noteBuffer_CUDA* noteBuffer, const uchar* keyState, const settings_CUDA* settings, const float* dynamicsProfile, const float* releaseProfile, uint* phazeWorkArr, uint* pressSamplessPassedWorkArr, uint* releaseSamplesPassedWorkArr, float* velocityWorkArr, const Generator_CUDA::soundFunctionPointer* soundFunction, synthesizer::generator_type currentGeneratorType){
    uint i = blockIdx.x * blockDim.x + threadIdx.x; // key index
    uint j = blockIdx.y * blockDim.y + threadIdx.y; // sample index
    if (i < settings->keyCount && j < settings->sampleSize){
        uint workArrIndex = j + i*settings->sampleSize;
        if (keyState[workArrIndex] || releaseSamplesPassedWorkArr[workArrIndex] < settings->release.duration){
            float dynamicsMultiplier = settings->dynamicsDuration > pressSamplessPassedWorkArr[workArrIndex] ? dynamicsProfile[pressSamplessPassedWorkArr[workArrIndex]] : settings->fadeTo;

            if (keyState[workArrIndex] == 0){
                dynamicsMultiplier *= releaseProfile[releaseSamplesPassedWorkArr[workArrIndex]];
                // printf("%d, %f, %d\n", workArrIndex, releaseProfile[releaseSamplesPassedWorkArr[workArrIndex]], releaseSamplesPassedWorkArr[workArrIndex]);
            }

            noteBuffer->buffer[workArrIndex] = (*soundFunction[currentGeneratorType])(phazeWorkArr[workArrIndex], noteBuffer->multiplier[i]) * settings->volume * velocityWorkArr[workArrIndex] * dynamicsMultiplier;
        }
    }
}

synthesizer::Generator_CUDA::Generator_CUDA(const settings_CUDA& settings){
    setGenerator(SINE);

    cudaMalloc((void**)(&d_phazeWorkArr), settings.keyCount * settings.sampleSize * sizeof(uint));
    cudaMalloc((void**)(&d_pressSamplessPassedWorkArr), settings.keyCount * settings.sampleSize * sizeof(uint));
    cudaMalloc((void**)(&d_releaseSamplesPassedWorkArr), settings.keyCount * settings.sampleSize * sizeof(uint));
    cudaMalloc((void**)(&d_velocityWorkArr), settings.keyCount * settings.sampleSize * sizeof(float));

    // cudaMalloc((void**)(&soundFunctions), (uint(synthesizer::LAST) + 1) * sizeof(*soundFunctions));
    cudaMalloc((void**)(&d_soundFunctions), (uint(synthesizer::LAST) + 1) * sizeof(soundFunctionPointer));
    kernel_initFunctionArray<<<1, 1>>>(d_soundFunctions);
}

synthesizer::Generator_CUDA::~Generator_CUDA(){
    cudaFree(d_phazeWorkArr);
    cudaFree(d_pressSamplessPassedWorkArr);
    cudaFree(d_releaseSamplesPassedWorkArr);
    cudaFree(d_velocityWorkArr);

    cudaFree(d_soundFunctions);
}

void synthesizer::Generator_CUDA::generate(noteBuffer_CUDA* d_noteBuffer, const uchar* d_keyState, const settings_CUDA* d_settings, const settings_CUDA& settings, const float* d_dynamicsProfile, const float* d_releaseProfile, const uint* d_releaseToAttackIndexMap){
    static const uint timeDependenciesBlockSize = 128;
    static const dim3 generateBlockSize(256, 1);

    uint timeDependenciesBlockCount = (settings.keyCount + timeDependenciesBlockSize - 1) / timeDependenciesBlockSize;
    dim3 generateBlockCount((settings.keyCount + generateBlockSize.x - 1) / generateBlockSize.x, (settings.sampleSize + generateBlockSize.y - 1) / generateBlockSize.y);

    kernel_preprocessTimeDependentValues<<<timeDependenciesBlockCount, timeDependenciesBlockSize>>>(d_noteBuffer, d_keyState, d_settings, d_releaseToAttackIndexMap, d_dynamicsProfile, d_phazeWorkArr, d_pressSamplessPassedWorkArr, d_releaseSamplesPassedWorkArr, d_velocityWorkArr);
    kernel_generate<<<generateBlockCount, generateBlockSize>>>(d_noteBuffer, d_keyState, d_settings, d_dynamicsProfile, d_releaseProfile, d_phazeWorkArr, d_pressSamplessPassedWorkArr, d_releaseSamplesPassedWorkArr, d_velocityWorkArr, d_soundFunctions, currentGeneratorType);
}

void synthesizer::Generator_CUDA::setGenerator(generator_type type){
    if (type == INVALID_GEN){
        return;
    }
    currentGeneratorType = type;
}

synthesizer::generator_type synthesizer::Generator_CUDA::getGeneratorType(){
    return currentGeneratorType;
}
