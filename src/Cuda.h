//
// Created by Grady Schofield on 10/7/23.
//

#ifndef STREAMING_LLAMA_CUDA_H
#define STREAMING_LLAMA_CUDA_H

#ifndef __APPLE__

class Cuda {
public:
    void allocateMemory();
    void launchKernel();
    void streamSynchronize();
};

#endif

#endif //STREAMING_LLAMA_CUDA_H
