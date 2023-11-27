#include "BufferConverter_Mono16.h"

void BufferConverter_Mono16::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
    static const uint maxValue = 0x00007FFF;
    int value;
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        value = maxValue * pipelineBuffer->bufferL[i];
        pcmBuffer->buff[j++] = value;
        pcmBuffer->buff[j++] = value >> 8;
    }
}
