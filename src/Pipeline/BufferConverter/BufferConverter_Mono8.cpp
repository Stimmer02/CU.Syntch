#include "BufferConverter_Mono8.h"

void BufferConverter_Mono8::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
    static const uint maxValue = 0x0000007F;
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        pcmBuffer->buff[j++] = maxValue * pipelineBuffer->buffer[i] + 127;
    }
}
