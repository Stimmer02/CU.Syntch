#include "BufferConverter_Mono16.h"

void BufferConverter_Mono16::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        pcmBuffer->buff[j++] = pipelineBuffer->buffer[i];
        pcmBuffer->buff[j++] = pipelineBuffer->buffer[i] >> 8;
    }
}
