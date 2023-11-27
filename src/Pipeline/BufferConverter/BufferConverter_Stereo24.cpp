#include "BufferConverter_Stereo24.h"

void BufferConverter_Stereo24::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
    static const uint maxValue = 0x007FFFFF;
    int valueL, valueR;
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        valueL = maxValue * pipelineBuffer->bufferL[i];
        valueR = maxValue * pipelineBuffer->bufferR[i];
        pcmBuffer->buff[j++] = valueL;
        pcmBuffer->buff[j++] = valueL >> 8;
        pcmBuffer->buff[j++] = valueL >> 16;
        pcmBuffer->buff[j++] = valueR;
        pcmBuffer->buff[j++] = valueR >> 8;
        pcmBuffer->buff[j++] = valueR >> 16;
    }
}
