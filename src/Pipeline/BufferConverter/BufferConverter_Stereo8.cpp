#include "BufferConverter_Stereo8.h"

void BufferConverter_Stereo8::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
    static const uint maxValue = 0x0000007F;
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        pcmBuffer->buff[j++] = maxValue * pipelineBuffer->bufferL[i] + 127;
        pcmBuffer->buff[j++] = maxValue * pipelineBuffer->bufferR[i] + 127;
    }
}
