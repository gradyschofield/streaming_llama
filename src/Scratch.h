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
    unique_ptr<Metal::MetalBuffer> metalBuffer;

public:
    Scratch(){
    }

    Scratch(int leadingDimension, int numColumns)
            : leadingDimension(leadingDimension),
              numColumns(numColumns),
              size(leadingDimension * numColumns * sizeof(T)),
              metalBuffer(make_unique<Metal::MetalBuffer>(size))
    {
    }

    T * getPtr() {
        return static_cast<T*>(metalBuffer->getMetalBuffer()->contents());
    }

    MTL::Buffer * getMetalBuffer() {
        return metalBuffer->getMetalBuffer();
    }

    size_t getSize() {
        return size;
    }

    int getLeadingDimension() {
        return leadingDimension;
    }
};

#endif //STREAMING_LLAMA_SCRATCH_H
