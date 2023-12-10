//
// Created by Grady Schofield on 10/7/23.
//

#ifndef STREAMING_LLAMA_CUDA_H
#define STREAMING_LLAMA_CUDA_H

#ifndef __APPLE__

#include<map>
#include<string>

#include<cuda.h>
#include<cublas_v2.h>
#include<mkl.h>

#include<Bf16.h>

using namespace std;

class Cuda {
    CUdevice device;
    CUcontext context;
    cublasHandle_t cublasHandle;
    map<string, CUmodule> modules;
    map<string, CUfunction> functions;

    void loadModule(string moduleName);
    void getFunction(string moduleName, string functionName);
public:
    Cuda();
    ~Cuda();
    void allocateMemory();
    void launchKernel();
    void streamSynchronize();
    void matmul(const enum CBLAS_TRANSPOSE TRANSA,
            const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
            const int K, const Bf16 ALPHA, const Bf16 * A, const int LDA,
            const Bf16 * B, const int LDB, const Bf16 BETA, Bf16 * C,
            const int LDC);
};

Cuda * getCuda();
void freeCuda();
void ce(CUresult result);

#endif

#endif //STREAMING_LLAMA_CUDA_H
