//
// Created by Grady Schofield on 10/1/23.
//

#ifndef STREAMING_LLAMA_WEIGHTS_H
#define STREAMING_LLAMA_WEIGHTS_H

#include<Common.h>

using namespace Common;

template<typename T>
class Weights {
    int64_t offsetIntoBlock;
    int numRows;
    int numColumns;
    int leadingDimension;

public:
    Weights() {
    }

    Weights(int64_t mapOffset, TensorFileInfo const & tfi)
            :   offsetIntoBlock(tfi.offset - mapOffset),
                numRows(tfi.numRows),
                numColumns(tfi.numColumns),
                leadingDimension(tfi.leadingDimension)
    {
    }

    T * getPtr(void* base) const {
        return (T*)((uint8_t*)base + offsetIntoBlock);
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
