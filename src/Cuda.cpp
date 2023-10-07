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
