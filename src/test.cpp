#include "Synthesizer_CUDA.h"
#include "Pipeline/BufferConverter/BufferConverter_Stereo16_CUDA.h"

#include <stdio.h>


int main(){
    const uint sampleSize = 10;
    const ushort keyCount = 10;

    uchar* buff[keyCount];
    for(int i = 0; i < keyCount; i++){
        buff[i] = new uchar[sampleSize];
        for (int j = 0; j < sampleSize; j++){
            buff[i][j] = 0;
        }
    }

    buff[0][2] = 1;
    buff[1][5] = 2;
    buff[2][0] = 3;
    buff[2][5] = 4;
    buff[3][1] = 5;
    buff[3][3] = 255;

    for (int i = 0; i < keyCount; i++){
        printf("key %d: ", i);
        for (int j = 0; j < sampleSize; j++){
            printf("\t%d ", buff[i][j]);
        }
        printf("\n");
    }

    keyboardTransferBuffer_CUDA* ktb = new keyboardTransferBuffer_CUDA(sampleSize, keyCount);

    ktb->convertBuffer(buff);

    uchar returnBuff[keyCount*sampleSize];
    cudaMemcpy(returnBuff, ktb->d_buffer, keyCount*sampleSize, cudaMemcpyDeviceToHost);

    audioFormatInfo afi;
    afi.sampleRate = 10;
    afi.channels = 2;
    afi.sampleSize = sampleSize;
    afi.bitDepth = 16;
    synthesizer::Synthesizer_CUDA* synth = new synthesizer::Synthesizer_CUDA(afi, keyCount);
        
    pipelineAudioBuffer_CUDA* audioBuffer = new pipelineAudioBuffer_CUDA(afi.sampleSize);

    synth->generateSample(audioBuffer, ktb);
    

    float* buffL = new float[afi.sampleSize];
    float* buffR = new float[afi.sampleSize];
    cudaMemcpy(buffL, audioBuffer->d_bufferL, afi.sampleSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(buffR, audioBuffer->d_bufferR, afi.sampleSize*sizeof(float), cudaMemcpyDeviceToHost);

    BufferConverter_Stereo16_CUDA* bc = new BufferConverter_Stereo16_CUDA(afi.sampleSize);
    struct::audioBuffer* pcmBuffer = new struct::audioBuffer(afi.sampleSize*afi.channels*2);

    printf("Audio buffer:\n");
    for (int i = 0; i < afi.sampleSize; i++){
        printf("%f,\t %f\n", buffL[i], buffR[i]);
    }

    bc->toPCM(audioBuffer, pcmBuffer);
    printf("PCM buffer:\n");
    for (int i = 0; i < afi.sampleSize*afi.channels*2; i++){
        printf("%d\n", pcmBuffer->buff[i]);
    }

    delete bc;
    delete pcmBuffer;
    delete[] buffL;
    delete[] buffR;
    for (int i = 0; i < keyCount; i++){
        delete[] buff[i];
    }
    delete audioBuffer;
    delete synth;
    delete ktb;
    return 0;
}