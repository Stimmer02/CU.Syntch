#ifndef COMPONENTSETTINGS_CUDA_H
#define COMPONENTSETTINGS_CUDA_H

#include <string>
#include <cuda_runtime.h>
#include <cuda.h>

typedef unsigned int uint;

namespace pipeline{
    struct componentSettings_CUDA{
    public:
        componentSettings_CUDA(const uint count, const std::string* names): count(count), names(names){
            values = new float[count];
            cudaMalloc(&d_values, count * sizeof(float));
        };
        ~componentSettings_CUDA(){
            delete[] values;
            cudaFree(d_values);
        }

        void copyToDevice(){
            cudaMemcpy(d_values, values, count * sizeof(float), cudaMemcpyHostToDevice);
        }

        const uint count;
        float* values;
        float* d_values;
        const std::string* names;
    };
}

#endif
