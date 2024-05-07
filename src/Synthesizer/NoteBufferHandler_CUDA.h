#ifndef NOTEBUFFERHANDLER_CUDA_H
#define NOTEBUFFERHANDLER_CUDA_H

#include "noteBuffer_CUDA.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace synthesizer{
    class NoteBufferHandler_CUDA{
    public:

        NoteBufferHandler_CUDA();
        NoteBufferHandler_CUDA(const uint& sampleSize, const uint& keyCount);
        ~NoteBufferHandler_CUDA();

        void init(const uint& sampleSize, const uint& keyCount);

        noteBuffer_CUDA* getDeviceNoteBuffer();

    private:
        void allocate(const uint& sampleSize, const uint& keyCount);
        void deallocate();

        uint keyCount;
        noteBuffer_CUDA* d_noteBuffer;
    };
}

#endif
