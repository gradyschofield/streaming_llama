//
// Created by Grady Schofield on 9/16/23.
//

#ifndef STREAMING_LLAMA_COMMON_H
#define STREAMING_LLAMA_COMMON_H

#include<cstdint>
#include<fstream>

using namespace std;

namespace Common {

    struct TensorFileInfo {
        int64_t offset;
        int numRows;
        int numColumns;
        int leadingDimension;
    };

    int findAlignment(int elements, int alignmentBytes) {
        int bytesPastAlignment = (elements * 4) % alignmentBytes;
        if(bytesPastAlignment == 0) return elements;
        else return (1 + ((elements * 4) / alignmentBytes)) * alignmentBytes / 4;
    }
}


#endif //STREAMING_LLAMA_COMMON_H
