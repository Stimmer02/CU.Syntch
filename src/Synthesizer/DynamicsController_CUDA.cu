#include <cuda_runtime_api.h>
#include <cuda.h>
#include "settings_CUDA.h"

using namespace synthesizer;

__global__ void kernel_calculateDynamicsProfileAttack(settings_CUDA* settings, float* noteDynamicsProfile){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < settings->attack.duration){
        noteDynamicsProfile[i] = i / float(settings->attack.duration);
    }
}

__global__ void kernel_calculateDynamicsProfileSustain(settings_CUDA* settings, float* noteDynamicsProfile, uint startIndex){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < settings->sustain.duration){
        noteDynamicsProfile[i + startIndex] = 1.0;
    }
}

__global__ void kernel_calculateDynamicsProfileFade(settings_CUDA* settings, float* noteDynamicsProfile, uint startIndex){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < settings->fade.duration){
        noteDynamicsProfile[i+startIndex] = (1 - i/float(settings->fade.duration)) * (1 - settings->fadeTo) + settings->fadeTo;
    }
}

__global__ void kernel_calculateReleaseProfileAndMap(uint newReleaseDuration, settings_CUDA* settings, float* noteReleaseProfile, uint* releaseToAttackIndexMap){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < newReleaseDuration){
        noteReleaseProfile[i] = 1 - i/float(newReleaseDuration);
        float ratio = settings->attack.duration / newReleaseDuration;
        releaseToAttackIndexMap[i] = (newReleaseDuration - 1 - i) * ratio;
    }
}