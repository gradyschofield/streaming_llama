#ifndef __APPLE__

#include<memory>
#include<mutex>

#include<cuda.h>

#include<Cuda.h>
#include<Exception.h>

using namespace std;

unique_ptr<Cuda> cudaPtr = nullptr;
mutex m;

Cuda::Cuda() {
    ce(cuInit(0));
    ce(cuGetDevice(&device, 0));
    size_t freeMem, totalMem;
    ce(cuMemGetInfo(&freeMem, & totalMem));
    cout << "Free GPU memory: " << freeMem << "\n";
    cout << "Total GPU memory: " << freeMem << "\n";
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