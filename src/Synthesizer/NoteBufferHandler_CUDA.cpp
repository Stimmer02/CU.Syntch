#include "NoteBufferHandler_CUDA.h"

using namespace synthesizer;

NoteBufferHandler_CUDA::NoteBufferHandler_CUDA(){
    cudaMalloc((void**)&d_noteBuffer, sizeof(noteBuffer_CUDA));
    keyCount = 0;
}

NoteBufferHandler_CUDA::NoteBufferHandler_CUDA(const uint& sampleSize, const uint& keyCount){
    cudaMalloc((void**)&d_noteBuffer, sizeof(noteBuffer_CUDA));
    allocate(sampleSize, keyCount);
}

NoteBufferHandler_CUDA::~NoteBufferHandler_CUDA(){
   deallocate();
   cudaFree(d_noteBuffer);
}

void NoteBufferHandler_CUDA::init(const uint& sampleSize, const uint& keyCount){
    deallocate();
    allocate(sampleSize, keyCount);
}

noteBuffer_CUDA* NoteBufferHandler_CUDA::getDeviceNoteBuffer(){
    return d_noteBuffer;
}

void NoteBufferHandler_CUDA::allocate(const uint& sampleSize, const uint& keyCount){
    this->keyCount = keyCount;

    noteBuffer_CUDA* h_noteBuffer = (noteBuffer_CUDA*)malloc(sizeof(noteBuffer_CUDA));

    cudaMalloc((void**)&(h_noteBuffer->buffer), keyCount * sampleSize * sizeof(float));

    cudaMalloc((void**)&(h_noteBuffer->phaze), keyCount * sizeof(uint));
    cudaMemset(&(h_noteBuffer->phaze), 0, keyCount * sizeof(uint));

    cudaMalloc((void**)&(h_noteBuffer->pressSamplessPassed), keyCount * sizeof(uint));
    cudaMemset(&(h_noteBuffer->pressSamplessPassed), 0, keyCount * sizeof(uint));

    cudaMalloc((void**)&(h_noteBuffer->lastKeyState), keyCount * sizeof(char));
    cudaMemset(&(h_noteBuffer->lastKeyState), 0, keyCount * sizeof(char));

    cudaMalloc((void**)&(h_noteBuffer->releaseSamplesPassed), keyCount * sizeof(uint));
    cudaMemset(&(h_noteBuffer->releaseSamplesPassed), 0, keyCount * sizeof(uint));

    cudaMalloc((void**)&(h_noteBuffer->stereoFactorL), keyCount * sizeof(float));
    cudaMemset(&(h_noteBuffer->stereoFactorL), 0, keyCount * sizeof(float));

    cudaMalloc((void**)&(h_noteBuffer->stereoFactorR), keyCount * sizeof(float));
    cudaMemset(&(h_noteBuffer->stereoFactorR), 0, keyCount * sizeof(float));

    cudaMalloc((void**)&(h_noteBuffer->frequency), keyCount * sizeof(float));
    cudaMemset(&(h_noteBuffer->frequency), 0, keyCount * sizeof(float));

    cudaMalloc((void**)&(h_noteBuffer->multiplier), keyCount * sizeof(float));
    cudaMemset(&(h_noteBuffer->multiplier), 0, keyCount * sizeof(float));

    cudaMalloc((void**)&(h_noteBuffer->velocity), keyCount * sizeof(float));
    cudaMemset(&(h_noteBuffer->velocity), 0, keyCount * sizeof(float));

    cudaMemcpy(d_noteBuffer, h_noteBuffer, sizeof(noteBuffer_CUDA), cudaMemcpyHostToDevice);

    free(h_noteBuffer);
}

void NoteBufferHandler_CUDA::deallocate(){
    if (this->keyCount != 0){
        void* temp;
        cudaMemcpy(&temp, &(d_noteBuffer->buffer), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaFree(temp);

        cudaMemcpy(&temp, &(d_noteBuffer->phaze), sizeof(uint*), cudaMemcpyDeviceToHost);
        cudaFree(temp);

        cudaMemcpy(&temp, &(d_noteBuffer->pressSamplessPassed), sizeof(uint*), cudaMemcpyDeviceToHost);
        cudaFree(temp);

        cudaMemcpy(&temp, &(d_noteBuffer->lastKeyState), sizeof(char*), cudaMemcpyDeviceToHost);
        cudaFree(temp);

        cudaMemcpy(&temp, &(d_noteBuffer->releaseSamplesPassed), sizeof(uint*), cudaMemcpyDeviceToHost);
        cudaFree(temp);

        cudaMemcpy(&temp, &(d_noteBuffer->stereoFactorL), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaFree(temp);

        cudaMemcpy(&temp, &(d_noteBuffer->stereoFactorR), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaFree(temp);

        cudaMemcpy(&temp, &(d_noteBuffer->frequency), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaFree(temp);

        cudaMemcpy(&temp, &(d_noteBuffer->multiplier), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaFree(temp);

        cudaMemcpy(&temp, &(d_noteBuffer->velocity), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaFree(temp);
    }
}