//
// Created by Grady Schofield on 10/6/23.
//

#include<Bf16.h>
#include<TransformerBlockScratch.h>

template<>
void allocateScratch<float>(size_t &totalAlloc, void ** p, int alignment, size_t size) {
    totalAlloc += size;
    posix_memalign(p, alignment, size);
}

template<>
void allocateScratch<Bf16>(size_t &totalAlloc, void ** p, int alignment, size_t size) {
    totalAlloc += size;
    posix_memalign(p, alignment, size);
}
