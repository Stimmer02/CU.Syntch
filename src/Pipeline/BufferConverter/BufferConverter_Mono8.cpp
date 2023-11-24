#include "BufferConverter_Mono8.h"

void BufferConverter_Mono8::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
    static const uint maxValue = 0x000000EF;
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        pcmBuffer->buff[j++] = maxValue * pipelineBuffer->buffer[i] + 127;//TODO: check if it works (looks really sus)
    }
}
