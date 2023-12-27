//
// Created by Grady Schofield on 10/1/23.
//

#ifndef STREAMING_LLAMA_SCRATCH_H
#define STREAMING_LLAMA_SCRATCH_H

#include<MetalHelpers.h>

template<typename T>
class Scratch {
    T * ptr;
    int leadingDimension;
    int numColumns;
    size_t size;
    shared_ptr<Metal::MetalBuffer> metalBuffer;

public:
    Scratch(){
    }

    template<typename Allocator>
    Scratch(Allocator && alignedAlloc, int alignmentBytes, int leadingDimension, int numColumns)
            : leadingDimension(leadingDimension),
              numColumns(numColumns),
              size(leadingDimension * numColumns * sizeof(T)),
              metalBuffer(make_shared<Metal::MetalBuffer>())
    {
        alignedAlloc((void**)&ptr, alignmentBytes, size);
        getMetalBuffer(ptr);
    }

    T * getPtr() {
        return ptr;
    }

    MTL::Buffer * getMetalBuffer(void * ptr) {
        return metalBuffer->getMetalBuffer(ptr, size);
    }

    size_t getSize() {
        return size;
    }

    int getLeadingDimension() {
        return leadingDimension;
    }
};

#endif //STREAMING_LLAMA_SCRATCH_H
