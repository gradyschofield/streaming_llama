//
// Created by Grady Schofield on 10/1/23.
//

#ifndef STREAMING_LLAMA_WEIGHTS_H
#define STREAMING_LLAMA_WEIGHTS_H

#include<cstdint>
#include<Common.h>
#include<MetalHelpers.h>

using namespace Common;

template<typename T>
class Weights {
    int64_t offsetIntoBlock;
    int numRows;
    int numColumns;
    int leadingDimension;
    shared_ptr<MetalHelpers::MetalBuffer> metalBuffer;

public:
    Weights() {
    }

    Weights & operator=(Weights const &) = delete;
    Weights & operator=(Weights &&) = default;
    Weights(Weights const &)= delete;
    Weights(Weights &&) = default;

    Weights(int64_t mapOffset, TensorFileInfo const & tfi)
            :   offsetIntoBlock(tfi.offset - mapOffset),
                numRows(tfi.numRows),
                numColumns(tfi.numColumns),
                leadingDimension(tfi.leadingDimension),
                metalBuffer(make_shared<MetalHelpers::MetalBuffer>())
    {
    }

    T * getPtr(void* base) {
        T * ptr = (T*)((uint8_t*)base + offsetIntoBlock);
        getMetalBuffer(ptr);
        return ptr;
    }

    MTL::Buffer * getMetalBuffer(void * ptr) {
        return metalBuffer->getMetalBuffer(ptr, getSize());
    }

    size_t getSize() const {
        return numRows * leadingDimension * sizeof(T);
    }

    int getNumRows() const {
        return numRows;
    }

    int getNumColumns() const {
        return numColumns;
    }

    int getLeadingDimension() const {
        return leadingDimension;
    }
};

#endif //STREAMING_LLAMA_WEIGHTS_H
