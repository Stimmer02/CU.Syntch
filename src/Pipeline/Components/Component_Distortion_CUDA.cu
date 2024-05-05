#include "Component_Distortion_CUDA.h"

using namespace pipeline;
const std::string Component_Distortion_CUDA::privateNames[4] = {"gain", "compress", "symmetry", "vol"};

__global__ void kernel_distortion(float* bufferR, float* bufferL, float* settings, uint size){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    float& gain = settings[0];
    float& compress = settings[1];
    float& symmetry = settings[2];
    float& vol = settings[3];

    if (i < size){
        float positiveGain = gain * (1 + symmetry);
        float negativeGain = gain * (1 - symmetry);

        if (bufferR[i] > 0){
            bufferR[i] *= positiveGain;
            if (bufferR[i] > compress){
                bufferR[i] = compress;
            }
        } else {
            bufferR[i] *= negativeGain;
            if (bufferR[i] < -compress){
                bufferR[i] = -compress;
            }
        }

        if (bufferL[i] > 0){
            bufferL[i] *= positiveGain;
            if (bufferL[i] > compress){
                bufferL[i] = compress;
            }
        } else {
            bufferL[i] *= negativeGain;
            if (bufferL[i] < -compress){
                bufferL[i] = -compress;
            }
        }

        bufferR[i] *= vol;
        bufferL[i] *= vol;
    }
}

Component_Distortion_CUDA::Component_Distortion_CUDA(const audioFormatInfo* audioInfo):AComponent_CUDA(audioInfo, 4, this->privateNames, COMP_DISTORION){
    defaultSettings();
}

Component_Distortion_CUDA::~Component_Distortion_CUDA(){

}

void Component_Distortion_CUDA::apply(pipelineAudioBuffer_CUDA* buffer){
    uint blockCount = (buffer->size + COMPONENT_BLOCK_SIZE - 1) / COMPONENT_BLOCK_SIZE;
    kernel_distortion<<<blockCount, COMPONENT_BLOCK_SIZE>>>(buffer->d_bufferR, buffer->d_bufferL, settings.d_values, buffer->size);
}

void Component_Distortion_CUDA::clear(){

}

void Component_Distortion_CUDA::defaultSettings(){
    settings.values[0] = 4.0;  //gain
    settings.values[1] = 0.5;  //compress
    settings.values[2] = 0.17; //symmetry
    settings.values[3] = 1.0;  //vol
    settings.copyToDevice();
}
