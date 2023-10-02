//
// Created by Grady Schofield on 10/1/23.
//

#ifndef STREAMING_LLAMA_SCRATCH_H
#define STREAMING_LLAMA_SCRATCH_H

template<typename T>
class Scratch {
    T * ptr;
    int leadingDimension;
public:
    Scratch(){
    }

    template<typename Allocator>
    Scratch(Allocator && alignedAlloc, int alignmentBytes, int leadingDimension, int numColumns)
            : leadingDimension(leadingDimension)
    {
        alignedAlloc((void**)&ptr, alignmentBytes, leadingDimension * numColumns * sizeof(T));
    }

    T * getPtr() {
        return ptr;
    }

    int getLeadingDimension() {
        return leadingDimension;
    }
};

#endif //STREAMING_LLAMA_SCRATCH_H
