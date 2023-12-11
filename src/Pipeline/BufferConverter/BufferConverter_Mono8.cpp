#include "BufferConverter_Mono8.h"
#include <immintrin.h>

void BufferConverter_Mono8::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
#ifdef AVX2
    static const uint maxValue = 0x0000007F;
    __m256 maxValueVec = _mm256_cvtepi32_ps(_mm256_set1_epi32(maxValue));
    __m256 input;
    __m256i temp, out;
    __m256i mask = _mm256_set_epi8(0x0C, 0x08, 0x04, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0C, 0x08, 0x04, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);

    __m256i permutation1 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 7, 3);
    __m256i permutation2 = _mm256_set_epi32(0, 0, 0, 0, 7, 3, 0, 0);
    __m256i permutation3 = _mm256_set_epi32(0, 0, 7, 3, 0, 0, 0, 0);
    __m256i permutation4 = _mm256_set_epi32(7, 3, 0, 0, 0, 0, 0, 0);

    __m256i increment = _mm256_set1_epi8(127);

    for (uint i = 0, j = 0; i < pipelineBuffer->size; j+=32){
        input = _mm256_load_ps(&pipelineBuffer->bufferL[i]);
        input = _mm256_mul_ps(input, maxValueVec);
        out = _mm256_cvtps_epi32(input);
        out = _mm256_shuffle_epi8(out, mask);
        out = _mm256_permutevar8x32_epi32(out, permutation1);
        i += 8;

        input = _mm256_load_ps(&pipelineBuffer->bufferL[i]);
        input = _mm256_mul_ps(input, maxValueVec);
        temp = _mm256_cvtps_epi32(input);
        temp = _mm256_shuffle_epi8(temp, mask);
        temp = _mm256_permutevar8x32_epi32(temp, permutation2);
        out = _mm256_or_si256(out, temp);
        i += 8;

        input = _mm256_load_ps(&pipelineBuffer->bufferL[i]);
        input = _mm256_mul_ps(input, maxValueVec);
        temp = _mm256_cvtps_epi32(input);
        temp = _mm256_shuffle_epi8(temp, mask);
        temp = _mm256_permutevar8x32_epi32(temp, permutation3);
        out = _mm256_or_si256(out, temp);
        i += 8;

        input = _mm256_load_ps(&pipelineBuffer->bufferL[i]);
        input = _mm256_mul_ps(input, maxValueVec);
        temp = _mm256_cvtps_epi32(input);
        temp = _mm256_shuffle_epi8(temp, mask);
        temp = _mm256_permutevar8x32_epi32(temp, permutation4);
        out = _mm256_or_si256(out, temp);
        i += 8;

        out = _mm256_add_epi8(out, increment);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pcmBuffer->buff[j]), out);

        //
    }
#else
    static const uint maxValue = 0x00007FFF;
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++, j++){
        pcmBuffer->buff[j] = maxValue * pipelineBuffer->bufferL[i] + 127;;
    }
#endif
}
