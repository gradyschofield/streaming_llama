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
    unique_ptr<Metal::MetalBuffer> metalBuffer;

public:
    Weights() {
    }

    Weights & operator=(Weights const &) = delete;
    Weights & operator=(Weights &&) = default;
    Weights(Weights const &)= delete;
    Weights(Weights &&) = default;

    Weights(void * mapAddress, int64_t mapOffset, TensorFileInfo const & tfi)
            :   offsetIntoBlock(tfi.offset - mapOffset),
                numRows(tfi.numRows),
                numColumns(tfi.numColumns),
                leadingDimension(tfi.leadingDimension),
                metalBuffer(make_unique<Metal::MetalBuffer>((uint8_t*)mapAddress + offsetIntoBlock, getSize()))
    {
    }

    T * getPtr() {
        return static_cast<T*>(metalBuffer->getMetalBuffer()->contents());
    }

    MTL::Buffer * getMetalBuffer() {
        return metalBuffer->getMetalBuffer();
    }

    size_t getSize() const {
        return numColumns * leadingDimension * sizeof(T);
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
