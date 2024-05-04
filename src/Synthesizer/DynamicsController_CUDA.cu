#include "DynamicsController_CUDA.h"
#include "stdio.h"

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

DynamicsController_CUDA::DynamicsController_CUDA(){
    dynamicsDuration = 0;
    dynamicsAllocated = 0;
    releaseDuration = 0;
    releaseAllocated = 0;
}

DynamicsController_CUDA::~DynamicsController_CUDA(){
    if (dynamicsAllocated > 0){
        cudaFree(d_noteDynamicsProfile);
    }
    if (releaseAllocated > 0){
        cudaFree(d_noteReleaseProfile);
        cudaFree(d_releaseToAttackIndexMap);
    }
}

void DynamicsController_CUDA::allocateDynamicsMemory(const settings_CUDA& settings){
    uint dynamicsFullLength = settings.attack.duration + settings.sustain.duration + settings.fade.duration;

    if (dynamicsFullLength == dynamicsDuration){
        return;
    }

    if (dynamicsFullLength > dynamicsAllocated){
        float* d_newNoteDynamicsProfile;
        float* d_oldNoteDynamicsProfile = d_noteDynamicsProfile;
        cudaMalloc((void**)&d_newNoteDynamicsProfile, dynamicsFullLength*sizeof(float));
        d_noteDynamicsProfile = d_newNoteDynamicsProfile;
        cudaFree(d_oldNoteDynamicsProfile);
        dynamicsAllocated = dynamicsFullLength;
    }

    dynamicsDuration = dynamicsFullLength;
}

void DynamicsController_CUDA::allocateReleaseMemory(const uint& newReleaseDuration){
    if (newReleaseDuration == releaseDuration){
        return;
    }

    if (newReleaseDuration > releaseAllocated){
        float* d_newNoteReleaseProfile;
        uint* d_newReleaseToAttackIndexMap;
        float* d_oldNoteReleaseProfile = d_noteReleaseProfile;
        uint* d_oldReleaseToAttackIndexMap = d_releaseToAttackIndexMap;
        cudaMalloc((void**)&d_newNoteReleaseProfile, newReleaseDuration*sizeof(float));
        cudaMalloc((void**)&d_newReleaseToAttackIndexMap, newReleaseDuration*sizeof(uint));
        d_noteReleaseProfile = d_newNoteReleaseProfile;
        d_releaseToAttackIndexMap = d_newReleaseToAttackIndexMap;
        cudaFree(d_oldNoteReleaseProfile);
        cudaFree(d_oldReleaseToAttackIndexMap);
        releaseAllocated = newReleaseDuration;
    }

    releaseDuration = newReleaseDuration;
}

void DynamicsController_CUDA::calculateDynamicsProfile(settings_CUDA* d_settings, settings_CUDA& settings){
    static const uint blockSize = 128;
    
    allocateDynamicsMemory(settings);

    uint sustainStartIndex = settings.attack.duration;
    uint fadeStartIndex = sustainStartIndex + settings.sustain.duration;

    uint attackNumBlocks = (settings.attack.duration + blockSize - 1) / blockSize;
    uint sustainNumBlocks = (settings.sustain.duration + blockSize - 1) / blockSize;
    uint fadeNumBlocks = (settings.fade.duration + blockSize - 1) / blockSize;

    kernel_calculateDynamicsProfileAttack<<<attackNumBlocks,blockSize>>>(d_settings, d_noteDynamicsProfile);
    kernel_calculateDynamicsProfileSustain<<<sustainNumBlocks,blockSize>>>(d_settings, d_noteDynamicsProfile, sustainStartIndex);
    kernel_calculateDynamicsProfileFade<<<fadeNumBlocks,blockSize>>>(d_settings, d_noteDynamicsProfile, fadeStartIndex);

    settings.dynamicsDuration = settings.attack.duration + settings.sustain.duration + settings.fade.duration;
    if (settings.fade.duration == 0){
        settings.fadeTo = 1;
    } else {
        settings.fadeTo = settings.rawFadeTo;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(&(d_settings->fadeTo), &(settings.fadeTo), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_settings->dynamicsDuration), &(settings.dynamicsDuration), sizeof(uint), cudaMemcpyHostToDevice);
}

void DynamicsController_CUDA::calculateReleaseProfile(settings_CUDA* d_settings, settings_CUDA& settings, float rawReleaseDuration){
    static const uint blockSize = 128;

    uint newReleaseDuration = settings.sampleRate * rawReleaseDuration;
    allocateReleaseMemory(newReleaseDuration);

    uint releaseNumBlocks = (newReleaseDuration + blockSize - 1) / blockSize;

    kernel_calculateReleaseProfileAndMap<<<releaseNumBlocks, blockSize>>>(newReleaseDuration, d_settings, d_noteReleaseProfile, d_releaseToAttackIndexMap);

    settings.release.duration = newReleaseDuration;
    settings.release.raw = rawReleaseDuration;
    cudaDeviceSynchronize();
    cudaMemcpy(d_settings, &settings, sizeof(settings_CUDA), cudaMemcpyHostToDevice);
}

void DynamicsController_CUDA::setDynamics(dynamics_CUDA* d_dynamics, dynamics_CUDA& dynamics, const float& raw, const uint& sampleRate){
    dynamics.raw = raw;
    dynamics.duration = sampleRate*raw;
    cudaMemcpy(d_dynamics, &dynamics, sizeof(dynamics_CUDA), cudaMemcpyHostToDevice);
}

const float* DynamicsController_CUDA::getDynamicsProfile(){
    return d_noteDynamicsProfile;
}

const float* DynamicsController_CUDA::getReleaseProfile(){
    return d_noteReleaseProfile;
}

const uint* DynamicsController_CUDA::getReleaseToAttackIndexMap(){
    return d_releaseToAttackIndexMap;
}

uint DynamicsController_CUDA::getDynamicsProfileLength(){
    return dynamicsDuration;
}

uint DynamicsController_CUDA::getReleaseProfileLength(){
    return releaseDuration;
}
