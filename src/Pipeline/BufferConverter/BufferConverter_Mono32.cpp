#include "BufferConverter_Mono32.h"

void BufferConverter_Mono32::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
    static const uint maxValue = 0x7FFFFFFF;
    int value;
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        value = maxValue * pipelineBuffer->bufferL[i];
        pcmBuffer->buff[j++] = value;
        pcmBuffer->buff[j++] = value >> 8;
        pcmBuffer->buff[j++] = value >> 16;
        pcmBuffer->buff[j++] = value >> 24;
    }
}
