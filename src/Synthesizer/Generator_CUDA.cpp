#include "Generator_CUDA.h"

synthesizer::Generator_CUDA::Generator_CUDA(const settings_CUDA& settings){
    d_currentGeneratorType = SINE;
    kernel_setGenerator<<<1, 1>>>(d_currentGeneratorType);

    cudaMalloc((void**)(&d_phazeWorkArr), settings.keyCount * settings.sampleSize * sizeof(uint));
    cudaMalloc((void**)(&d_pressSamplessPassedWorkArr), settings.keyCount * settings.sampleSize * sizeof(uint));
    cudaMalloc((void**)(&d_releaseSamplesPassedWorkArr), settings.keyCount * settings.sampleSize * sizeof(uint));
    cudaMalloc((void**)(&d_velocityWorkArr), settings.keyCount * settings.sampleSize * sizeof(float));
}

synthesizer::Generator_CUDA::~Generator_CUDA(){
    cudaFree(d_phazeWorkArr);
    cudaFree(d_pressSamplessPassedWorkArr);
    cudaFree(d_releaseSamplesPassedWorkArr);
    cudaFree(d_velocityWorkArr);
}

void synthesizer::Generator_CUDA::generate(noteBuffer_CUDA* d_noteBuffer, const uchar* d_keyState, const settings_CUDA* d_settings, const settings_CUDA& settings, const float* d_dynamicsProfile, const float* d_releaseProfile, const uint* d_releaseToAttackIndexMap){
    static const uint timeDependenciesBlockSize = 128;
    static const dim3 generateBlockSize(256, 1);

    uint timeDependenciesBlockCount = (settings.keyCount + timeDependenciesBlockSize - 1) / timeDependenciesBlockSize;
    dim3 generateBlockCount((settings.keyCount + generateBlockSize.x - 1) / generateBlockSize.x, (settings.sampleSize + generateBlockSize.y - 1) / generateBlockSize.y);

    kernel_preprocessTimeDependentValues<<<timeDependenciesBlockCount, timeDependenciesBlockSize>>>(d_noteBuffer, d_keyState, d_settings);
    kernel_generate<<<generateBlockCount, generateBlockSize>>>(d_noteBuffer, d_keyState, d_settings, d_dynamicsProfile, d_releaseProfile);
}

void synthesizer::Generator_CUDA::setGenerator(generator_type type){
    if (type == INVALID_GEN){
        return;
    }
    d_currentGeneratorType = type;
    kernel_setGenerator<<<1, 1>>>(d_currentGeneratorType);
}

synthesizer::generator_type synthesizer::Generator_CUDA::getGeneratorType(){
    return d_currentGeneratorType;
}
