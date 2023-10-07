//
// Created by Grady Schofield on 10/7/23.
//

#ifndef STREAMING_LLAMA_CUDA_H
#define STREAMING_LLAMA_CUDA_H

#ifndef __APPLE__

#include<cuda.h>

class Cuda {
    CUdevice device;
public:
    Cuda();
    void allocateMemory();
    void launchKernel();
    void streamSynchronize();
};

Cuda * getCuda();
void ce(CUresult result);

#endif

#endif //STREAMING_LLAMA_CUDA_H
