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
    getFunction(module, "swilu");
    getFunction(module, "rope");
    getFunction(module, "layerNorm");

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
        cublasDestroy(cublasHandle);
        for(auto & p : modules) {
            CUmodule module = p.second;
            ce(cuModuleUnload(module));
        }
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

void ce(CUresult result) {
    if(result != CUDA_SUCCESS) {
        char const * str;
        cuGetErrorString(result, &str);
        throw Exception(string(str));
    }
}

#endif
