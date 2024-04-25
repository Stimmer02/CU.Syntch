#include "UserInput/keyboardTransferBuffer_CUDA.h"

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
        printf("Key %d: ", i);
        for (uint j = 0; j < sampleSize; j++){
            printf("%d\t", buff[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    keyboardTransferBuffer_CUDA* ktb = new keyboardTransferBuffer_CUDA(sampleSize, keyCount);

    ktb->convertBuffer(buff);

    uchar returnBuff[keyCount*sampleSize];
    cudaMemcpy(returnBuff, ktb->d_buffer, keyCount*sampleSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < keyCount; i++){
        printf("Key %d: ", i);
        for (uint j = 0; j < sampleSize; j++){
            printf("%d\t", returnBuff[i*sampleSize + j]);
        }
        printf("\n");
    }

    delete ktb;
    return 0;
}