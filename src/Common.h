//
// Created by Grady Schofield on 9/16/23.
//

#ifndef STREAMING_LLAMA_COMMON_H
#define STREAMING_LLAMA_COMMON_H

#include<cstdint>

struct TensorFileInfo {
    uint64_t offset;
    int numRows;
    int numColumns;
    int leadingDimension;
};


#endif //STREAMING_LLAMA_COMMON_H
