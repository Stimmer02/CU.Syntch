#include "BufferConverter_Mono16.h"


void BufferConverter_Mono16::toPCM(pipelineAudioBuffer* pipelineBuffer, audioBuffer* pcmBuffer){
#ifdef AVX2
    static const uint maxValue = 0x00007FFF;
    __m256i out, move, tempConverted, permutation1, permutation2;
    __m256 maxValueVec, input, temp;

    permutation1 = _mm256_set_epi32(3, 2, 5, 4, 7, 6, 1, 0);
    permutation2 = _mm256_set_epi32(7, 6, 1, 0, 3, 2, 5, 4);

    move = _mm256_set_epi8(0x0d, 0x0c, 0x09, 0x08, 0x05, 0x04, 0x01, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0d, 0x0c, 0x09, 0x08, 0x05, 0x04, 0x01, 0x00);

    maxValueVec = _mm256_cvtepi32_ps(_mm256_set1_epi32(maxValue));
    for (uint i = 0, j = 0; i < pipelineBuffer->size; j+=32){
        input = _mm256_load_ps(&pipelineBuffer->bufferL[i]);
        temp = _mm256_mul_ps(maxValueVec, input);
        out = _mm256_cvtps_epi32(temp);
        out = _mm256_shuffle_epi8(out, move);
        out = _mm256_permutevar8x32_epi32(out, permutation1);
        i += 8;

        input = _mm256_load_ps(&pipelineBuffer->bufferL[i]);
        temp = _mm256_mul_ps(maxValueVec, input);
        tempConverted = _mm256_cvtps_epi32(temp);
        tempConverted = _mm256_shuffle_epi8(tempConverted, move);
        tempConverted = _mm256_permutevar8x32_epi32(tempConverted, permutation2);
        out = _mm256_or_si256(out, tempConverted);
        i += 8;

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pcmBuffer->buff[j]), out);
    }
#else
    static const uint maxValue = 0x00007FFF;
    int value;
    for (uint i = 0, j = 0; i < pipelineBuffer->size; i++){
        value = maxValue * pipelineBuffer->bufferL[i];
        printf("0x%04X; %d; %f\n", value, maxValue, pipelineBuffer->bufferL[i]);
        pcmBuffer->buff[j++] = value;
        pcmBuffer->buff[j++] = value >> 8;
    }
#endif
}
