#include "BufferConverter_Mono24.h"

void BufferConverter_Mono24::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        pcmBuffer->buff[j++] = pipelineBuffer->buffer[i];
        pcmBuffer->buff[j++] = pipelineBuffer->buffer[i] >> 8;
        pcmBuffer->buff[j++] = pipelineBuffer->buffer[i] >> 16;
    }
}
