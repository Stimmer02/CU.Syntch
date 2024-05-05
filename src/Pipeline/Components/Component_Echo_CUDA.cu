#include "Component_Echo_CUDA.h"

using namespace pipeline;
const std::string Component_Echo_CUDA::privateNames[5] = {"lvol", "rvol", "delay", "fade", "repeats"};

__global__ void kernel_echoCopy(float* bufferR, float* bufferL, float* lMemory, float* rMemory, uint size, uint memoryShift){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size){
        rMemory[memoryShift + i] = bufferR[i];
        lMemory[memoryShift + i] = bufferL[i];
    }
}

__global__ void kernel_echoSum(float* bufferR, float* bufferL, float* lMemory, float* rMemory, float lVolume, float rVolume, uint size, uint generalIndexShift, uint memoryShift){
    uint i = blockIdx.x * blockDim.x + threadIdx.x + generalIndexShift;
    if (i < size){
        bufferR[i] += rMemory[memoryShift + i] * rVolume;
        bufferL[i] += lMemory[memoryShift + i] * lVolume;
    }
}

Component_Echo_CUDA::Component_Echo_CUDA(const audioFormatInfo* audioInfo):AComponent_CUDA(audioInfo, 5, this->privateNames, COMP_ECHO), maxDelayTime(10){
    currentSample = 0;
    sampleCount = audioInfo->sampleRate * maxDelayTime / audioInfo->sampleSize;
    sampleCount += 1 ? 0 : (audioInfo->sampleRate * maxDelayTime % audioInfo->sampleSize) > 0;

    cudaMalloc(&d_lMemory, audioInfo->sampleSize * sampleCount * sizeof(float));
    cudaMalloc(&d_rMemory, audioInfo->sampleSize * sampleCount * sizeof(float));

    defaultSettings();
    clear();
}

Component_Echo_CUDA::~Component_Echo_CUDA(){
    cudaFree(d_lMemory);
    cudaFree(d_rMemory);
}

void Component_Echo_CUDA::apply(pipelineAudioBuffer_CUDA* buffer){
    uint blockCount = (buffer->size + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    
    uint memoryShift = currentSample * audioInfo->sampleSize;
    kernel_echoCopy<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, d_lMemory, d_rMemory, buffer->size, memoryShift);

    uint allSamplesShift = audioInfo->sampleRate * delay;
    int sampleIndex = currentSample - allSamplesShift / audioInfo->sampleSize - 1;
    if (sampleIndex < 0){
        sampleIndex = sampleCount + sampleIndex;
    }
    uint singleSampleShift = allSamplesShift % audioInfo->sampleSize;
    uint indexShift = audioInfo->sampleSize - singleSampleShift;

    float rVolume = settings.values[1];
    float lVolume = settings.values[0];

    memoryShift = sampleIndex * audioInfo->sampleSize + indexShift;
    blockCount = (singleSampleShift + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    kernel_echoSum<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, d_lMemory, d_rMemory, lVolume, rVolume, singleSampleShift, 0, memoryShift);

    sampleIndex++;
    if (sampleIndex == sampleCount){
        sampleIndex = 0;
    }

    memoryShift = sampleIndex * audioInfo->sampleSize - singleSampleShift;
    blockCount = (audioInfo->sampleSize + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    kernel_echoSum<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, d_lMemory, d_rMemory, lVolume, rVolume, audioInfo->sampleSize, singleSampleShift, memoryShift);

    for (uint repeat = 1; repeat < repeats; repeat++){
        allSamplesShift = audioInfo->sampleRate * delay * repeat;
        sampleIndex = currentSample - allSamplesShift / audioInfo->sampleSize - 1;
        if (sampleIndex < 0){
            sampleIndex = sampleCount + sampleIndex;
        }
        singleSampleShift = allSamplesShift % audioInfo->sampleSize;
        indexShift = audioInfo->sampleSize - singleSampleShift;

        rVolume *= fade;
        lVolume *= fade;

        memoryShift = sampleIndex * audioInfo->sampleSize + indexShift;
        blockCount = (singleSampleShift + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
        kernel_echoSum<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, d_lMemory, d_rMemory, lVolume, rVolume, singleSampleShift, 0, memoryShift);

        sampleIndex++;
        if (sampleIndex == sampleCount){
            sampleIndex = 0;
        }

        memoryShift = sampleIndex * audioInfo->sampleSize - singleSampleShift;
        blockCount = (audioInfo->sampleSize + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
        kernel_echoSum<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, d_lMemory, d_rMemory, lVolume, rVolume, audioInfo->sampleSize, singleSampleShift, memoryShift);
    }



    currentSample++;
    if (currentSample == sampleCount){
        currentSample = 0;
    }
}

void Component_Echo_CUDA::clear(){
    cudaMemset(d_lMemory, 0, audioInfo->sampleSize * sampleCount * sizeof(float));
    cudaMemset(d_rMemory, 0, audioInfo->sampleSize * sampleCount * sizeof(float));
}

void Component_Echo_CUDA::defaultSettings(){
    settings.values[0] = 0.5;  //lvol
    settings.values[1] = 0.5;  //rvol
    settings.values[2] = 0.2;  //delay
    settings.values[3] = 0.7;  //fade
    settings.values[4] = 5;    //repeats
    settings.copyToDevice();
}

void Component_Echo_CUDA::set(uint index, float value){
    switch (index) {
        case 2:
            if (value > maxDelayTime){
                value = maxDelayTime;
            } else if (value < 0){
                value = 0;
            }
        case 4:
            if (value * delay > maxDelayTime){
                value = maxDelayTime / delay;
            } else if (value < 1){
                value = 1;
            }
    }

    settings.values[index] = value;
    settings.copyToDevice();
}