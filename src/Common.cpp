//
// Created by Grady Schofield on 9/22/23.
//

#include<Common.h>

//ofstream logger("log");

namespace Common {
    int findAlignment(int elements, int alignmentBytes) {
        int bytesPastAlignment = (elements * 4) % alignmentBytes;
        if (bytesPastAlignment == 0) return elements;
        else return (1 + ((elements * 4) / alignmentBytes)) * alignmentBytes / 4;
    }
}