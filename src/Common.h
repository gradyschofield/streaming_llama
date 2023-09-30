//
// Created by Grady Schofield on 9/16/23.
//

#ifndef STREAMING_LLAMA_COMMON_H
#define STREAMING_LLAMA_COMMON_H

#include<cstdint>
#include<fstream>

using namespace std;

//extern ofstream logger;

namespace Common {

    enum FileStorageFormat {
        Fp32Aligned,
        Bf16Aligned,
        Cuda
    };

    uint8_t fileStorageFormatToInt(FileStorageFormat fileStorageFormat);
    FileStorageFormat intToFileStorageFormat(uint8_t i);

    struct TensorFileInfo {
        int64_t offset;
        int numRows;
        int numColumns;
        int leadingDimension;
    };

    int findAlignment(int elements, int alignmentBytes);
}


#endif //STREAMING_LLAMA_COMMON_H
