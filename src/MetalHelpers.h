//
// Created by Grady Schofield on 12/10/23.
//

#ifndef STREAMING_LLAMA_METALHELPERS_H
#define STREAMING_LLAMA_METALHELPERS_H

#include<Metal.hpp>

#include<iostream>

using namespace std;

namespace MetalHelpers {
    MTL::Device * getDevice();
    MTL::Buffer * newBuffer(void * p, long len);
    void releaseBuffer(MTL::Buffer * buffer);
    MTL::Buffer * getBuffer(void const * p);

    class MetalBuffer {
        MTL::Buffer * buffer = nullptr;
        void * ptr = nullptr;

    public:
        MTL::Buffer * getMetalBuffer(void * ptr, size_t size) {
            if (!buffer) {
                buffer = MetalHelpers::newBuffer(ptr, size);
                this->ptr = ptr;
            } else if (ptr != this->ptr) {
                MetalHelpers::releaseBuffer(buffer);
                buffer = MetalHelpers::newBuffer(ptr, size);
                this->ptr = ptr;
                cout << "Warning, pointer for metal buffer changed" << endl;
            }
            return buffer;
        }

        ~MetalBuffer() {
            MetalHelpers::releaseBuffer(buffer);
        }
    };
}

#endif //STREAMING_LLAMA_METALHELPERS_H
