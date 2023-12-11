#include "BufferConverter_Mono32.h"

void BufferConverter_Mono32::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
#ifdef AVX2
    static const uint maxValue = 0x7FFFFF80;
    __m256i out;
    __m256 maxValueVec, input, temp;

    maxValueVec = _mm256_cvtepi32_ps(_mm256_set1_epi32(maxValue));
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i+=8, j+=32){
        input = _mm256_load_ps(&pipelineBuffer->bufferL[i]);
        temp = _mm256_mul_ps(maxValueVec, input);
        out = _mm256_cvtps_epi32(temp);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pcmBuffer->buff[j]), out);
    }
#else
    static const uint maxValue = 0x7FFFFF80;
    int value;
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        value = maxValue * pipelineBuffer->bufferL[i];
        pcmBuffer->buff[j++] = value;
        pcmBuffer->buff[j++] = value >> 8;
        pcmBuffer->buff[j++] = value >> 16;
        pcmBuffer->buff[j++] = value >> 24;
    }
#endif
}
