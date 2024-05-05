#include "Component_SimpleCompressor_CUDA.h"

using namespace pipeline;
const std::string Component_SimpleCompressor_CUDA::privateNames[3] = {"threshold", "ratio", "vol"};



__global__ void kernel_simpleCompressor(float* bufferR, float* bufferL, float* settings, uint size){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    float& threshold = settings[0];
    float& ratio = settings[1];
    float& vol = settings[2];

    if (i < size){
        float rAbs = abs(bufferR[i]);
        float lAbs = abs(bufferL[i]);
        
        if (rAbs > threshold){
            bufferR[i] = copysignf(threshold + (rAbs - threshold) / ratio, bufferR[i]);
        }

        if (lAbs > threshold){
            bufferL[i] = copysignf(threshold + (lAbs - threshold) / ratio, bufferL[i]);
        }

        bufferR[i] *= vol;
        bufferL[i] *= vol;

    }
}


Component_SimpleCompressor_CUDA::Component_SimpleCompressor_CUDA(const audioFormatInfo* audioInfo):AComponent_CUDA(audioInfo, 3, this->privateNames, COMP_COMPRESSOR){
    defaultSettings();
    clear();
}

Component_SimpleCompressor_CUDA::~Component_SimpleCompressor_CUDA(){

}

void Component_SimpleCompressor_CUDA::apply(pipelineAudioBuffer_CUDA* buffer){
    uint blockCount = (audioInfo->sampleSize + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    
    kernel_simpleCompressor<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, settings.d_values, audioInfo->sampleSize);
}

void Component_SimpleCompressor_CUDA::clear(){

}

void Component_SimpleCompressor_CUDA::defaultSettings(){
    settings.values[0] = 0.8;     //threshold
    settings.values[1] = 10;     //ratio
    settings.values[2] = 1.0;     //vol

    settings.copyToDevice();
}

void Component_SimpleCompressor_CUDA::set(uint index, float value){
    switch (index) {
        case 0:
            if (value < 0){
                value = 0;
            }
            break;
    }

    settings.values[index] = value;
    settings.copyToDevice();
}
