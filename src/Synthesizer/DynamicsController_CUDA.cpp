#include "DynamicsController_CUDA.h"
#include "DynamicsController_CUDA.cu"


using namespace synthesizer;

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
        cudaFree(d_noteDynamicsProfile);
        cudaMalloc((void**)&d_noteDynamicsProfile, dynamicsFullLength*sizeof(float));
        dynamicsAllocated = dynamicsFullLength;
    }

    dynamicsDuration = dynamicsFullLength;
}

void DynamicsController_CUDA::allocateReleaseMemory(const uint& newReleaseDuration){
    if (newReleaseDuration == releaseDuration){
        return;
    }

    if (newReleaseDuration > releaseAllocated){
        cudaFree(d_noteReleaseProfile);
        cudaMalloc((void**)&d_noteReleaseProfile, newReleaseDuration*sizeof(float));
        cudaMalloc((void**)&d_releaseToAttackIndexMap, newReleaseDuration*sizeof(uint));
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

    kernel_calculateDynamicsProfileAttack<<<attackNumBlocks,blockSize>>>(d_noteDynamicsProfile, d_settings);
    kernel_calculateDynamicsProfileSustain<<<sustainNumBlocks,blockSize>>>(d_noteDynamicsProfile, d_settings, sustainStartIndex);
    kernel_calculateDynamicsProfileFade<<<sustainNumBlocks,blockSize>>>(d_noteDynamicsProfile, d_settings, fadeStartIndex);

    settings.dynamicsDuration = settings.attack.duration + settings.sustain.duration + settings.fade.duration;
    if (settings.fade.duration == 0){
        settings.fadeTo = 1;
    } else {
        settings.fadeTo = settings.rawFadeTo;
    }

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
