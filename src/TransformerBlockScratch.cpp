//
// Created by Grady Schofield on 10/6/23.
//

#include<Bf16.h>
#include<TransformerBlockScratch.h>

template<>
void allocateScratch<float, Cpu>(size_t &totalAlloc, void ** p, int alignment, size_t size) {
    totalAlloc += size;
    posix_memalign(p, alignment, size);
}

template<>
void allocateScratch<Bf16, Cpu>(size_t &totalAlloc, void ** p, int alignment, size_t size) {
    totalAlloc += size;
    posix_memalign(p, alignment, size);
}

#ifdef BUILD_CUDA
template<>
void allocateScratch<float, Cuda>(size_t &totalAlloc, void ** p, int alignment, size_t size) {
    totalAlloc += size;
    Cudeviceptr **ptr = (Cudeviceptr**)p;
    *ptr = new Cudeviceptr;
    ce(cuMemAlloc(*p, ));
}

template<>
void allocateScratch<Bf16, Cuda>(size_t &totalAlloc, void ** p, int alignment, size_t size) {
    totalAlloc += size;
    ce(cuMemAlloc(*(Cudeviceptr**)p, ));
}
#endif