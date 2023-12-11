#include "BufferConverter_Mono24.h"
#include <immintrin.h>

#include <iostream>

void BufferConverter_Mono24::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
#ifdef AVX2
    static const uint maxValue = 0x007FFFFF;
    __m256i out, move, permutation;
    __m256 maxValueVec, input, temp;
    unsigned char lastIteration[32];

    permutation = _mm256_set_epi32(0, 4, 5, 6, 7, 1, 2, 3);
    move = _mm256_set_epi8(4, 2, 1, 0, 9, 8, 6, 5, 14, 13, 12, 10, -1, -1, -1, -1, 4, 2, 1, 0, 9, 8, 6, 5, 14, 13, 12, 10, -1, -1, -1, -1);

    maxValueVec = _mm256_cvtepi32_ps(_mm256_set1_epi32(maxValue));
    uint i = 0, j = 0;
    for (; i < pipelineBuffer->size-8; i+=8, j+=24){
        input = _mm256_load_ps(&pipelineBuffer->bufferL[i]);
        temp = _mm256_mul_ps(maxValueVec, input);
        out = _mm256_cvtps_epi32(temp);
        out = _mm256_shuffle_epi8(out, move);
        out = _mm256_permutevar8x32_epi32(out, permutation);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pcmBuffer->buff[j]), out);
    }
    input = _mm256_load_ps(&pipelineBuffer->bufferL[i]);
    temp = _mm256_mul_ps(maxValueVec, input);
    out = _mm256_cvtps_epi32(temp);
    out = _mm256_shuffle_epi8(out, move);
    out = _mm256_permutevar8x32_epi32(out, permutation);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&lastIteration), out);
    std::memcpy(&pcmBuffer->buff[j], lastIteration, 24);


#else
    static const uint maxValue = 0x007FFFFF;
    int value;
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        value = maxValue * pipelineBuffer->bufferL[i];
        pcmBuffer->buff[j++] = value;
        pcmBuffer->buff[j++] = value >> 8;
        pcmBuffer->buff[j++] = value >> 16;
    }
#endif
}

