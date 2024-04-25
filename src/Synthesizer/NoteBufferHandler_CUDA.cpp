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

    cudaMalloc((void**)&(d_noteBuffer->buffer), keyCount * sampleSize * sizeof(float));

    cudaMalloc((void**)&(d_noteBuffer->phaze), keyCount * sizeof(uint));
    cudaMemset(&(d_noteBuffer->phaze), 0, keyCount * sizeof(uint));

    cudaMalloc((void**)&(d_noteBuffer->pressSamplessPassed), keyCount * sizeof(uint));
    cudaMemset(&(d_noteBuffer->pressSamplessPassed), 0, keyCount * sizeof(uint));

    cudaMalloc((void**)&(d_noteBuffer->lastKeyState), keyCount * sizeof(char));
    cudaMemset(&(d_noteBuffer->lastKeyState), 0, keyCount * sizeof(char));

    cudaMalloc((void**)&(d_noteBuffer->releaseSamplesPassed), keyCount * sizeof(uint));
    cudaMemset(&(d_noteBuffer->releaseSamplesPassed), 0, keyCount * sizeof(uint));

    cudaMalloc((void**)&(d_noteBuffer->stereoFactorL), keyCount * sizeof(float));
    cudaMemset(&(d_noteBuffer->stereoFactorL), 0, keyCount * sizeof(float));

    cudaMalloc((void**)&(d_noteBuffer->stereoFactorR), keyCount * sizeof(float));
    cudaMemset(&(d_noteBuffer->stereoFactorR), 0, keyCount * sizeof(float));

    cudaMalloc((void**)&(d_noteBuffer->frequency), keyCount * sizeof(float));
    cudaMemset(&(d_noteBuffer->frequency), 0, keyCount * sizeof(float));

    cudaMalloc((void**)&(d_noteBuffer->multiplier), keyCount * sizeof(float));
    cudaMemset(&(d_noteBuffer->multiplier), 0, keyCount * sizeof(float));

    cudaMalloc((void**)&(d_noteBuffer->velocity), keyCount * sizeof(float));
    cudaMemset(&(d_noteBuffer->velocity), 0, keyCount * sizeof(float));
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