#ifndef __APPLE__

#include<iostream>
#include<memory>
#include<mutex>

#include<cuda.h>
#include<cublas_v2.h>

#include<Cuda.h>
#include<Exception.h>

using namespace std;

unique_ptr<Cuda> cudaPtr = nullptr;
mutex m;

Cuda::Cuda() {
    ce(cuInit(0));
    int deviceCount;
    ce(cuDeviceGetCount(&deviceCount));
    if(deviceCount < 1) {
        throw Exception("No cuda devices found on this computer.");
    }
    ce(cuDeviceGet(&device, 0));
    ce(cuCtxCreate(&context, 0, device));

    loadModule("swilu.ptx");
    loadModule("rope.ptx");
    loadModule("layerNorm.ptx");
    getFunction("swilu.ptx", "swilu");
    getFunction("rope.ptx", "rope");
    getFunction("layerNorm.ptx", "layerNorm");

    size_t freeMem, totalMem;
    ce(cuMemGetInfo(&freeMem, & totalMem));
    cublasCreate(&cublasHandle);
    cout << "Free GPU memory: " << freeMem << "\n";
    cout << "Total GPU memory: " << freeMem <<endl;
}

void Cuda::loadModule(string moduleName) {
    CUmodule module;
    ce(cuModuleLoad(&module, moduleName.c_str()));
    modules.emplace(moduleName, module);
}

void Cuda::getFunction(string moduleName, string functionName) {
    CUfunction function;
    CUmodule module = mapAt(modules, moduleName);
    ce(cuModuleGetFunction(&function, module, functionName.c_str()));
    functions.emplace(functionName, function);
}

Cuda::~Cuda() {
    /*
     * free memory with cuMemFree(devpointer)
     * destroy events with cuEventDestroy(ev)
     */
    try {
        for(auto & p : modules) {
            cout << "unloading " << p.first << " " << p.second << endl;
            CUmodule module = p.second;
            ce(cuModuleUnload(module));
        }
        cublasDestroy(cublasHandle);
        cuCtxDestroy(context);
    } catch(Exception & e) {
        cout << e.what() << endl;
    }
}

Cuda * getCuda() {
    lock_guard<mutex> lg(m);
    if (!cudaPtr) {
        cudaPtr = make_unique<Cuda>();
    }
    return cudaPtr.get();
}

void freeCuda() {
    lock_guard<mutex> lg(m);
    if (cudaPtr) {
        cudaPtr.reset(nullptr);
    }
}

void ce(CUresult result) {
    if(result != CUDA_SUCCESS) {
        char const * str;
        cuGetErrorString(result, &str);
        stringstream sstr;
        sstr << "Cuda error enum " << result << ": " << str;
        throw Exception(sstr.str());
    }
}

void Cuda::matmul(const enum CBLAS_TRANSPOSE TRANSA,
        const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
        const int K, const Bf16 ALPHA, const Bf16 * A, const int LDA,
        const Bf16 * B, const int LDB, const Bf16 BETA, Bf16 * C,
        const int LDC) {
    CUevent ev1, ev2;
    cuEventRecord(ev1, 0);
    cublasOperation_t transa = TRANSA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transb = TRANSB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasGemmEx(cublasHandle, transa, transb, M, N, K,
                 &ALPHA, A, CUDA_R_16BF, LDA,
                 B, CUDA_R_16BF, LDB, &BETA, C, CUDA_R_16BF, LDC,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cuEventRecord(ev2, 0);
    cuEventSynchronize(ev2);
    float millis;
    cuEventElapsedTime(&millis, ev1, ev2);
}

#endif
