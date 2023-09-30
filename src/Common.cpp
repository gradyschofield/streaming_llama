//
// Created by Grady Schofield on 9/22/23.
//

#include<Common.h>

//ofstream logger("log");

namespace Common {
    uint8_t fileStorageFormatToInt(FileStorageFormat outputFormat) {
        switch(outputFormat) {
            case Bf16Aligned:
                return 0;
            case Fp32Aligned:
                return 1;
            case Cuda:
                return 2;
        }
    }

    FileStorageFormat intToFileStorageFormat(uint8_t i) {
        switch(i) {
            case 0:
                return Bf16Aligned;
            case 1:
                return Fp32Aligned;
            case 2:
                return Cuda;
            default:
                throw 1;
        }
    }

    int findAlignment(int elements, int alignmentBytes) {
        int bytesPastAlignment = (elements * 4) % alignmentBytes;
        if (bytesPastAlignment == 0) return elements;
        else return (1 + ((elements * 4) / alignmentBytes)) * alignmentBytes / 4;
    }
}