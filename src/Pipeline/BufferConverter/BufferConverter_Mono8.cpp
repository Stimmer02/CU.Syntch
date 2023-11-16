#include "BufferConverter_Mono8.h"

void BufferConverter_Mono8::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        pcmBuffer->buff[j++] = pipelineBuffer->buffer[i] + 127;
    }
}
