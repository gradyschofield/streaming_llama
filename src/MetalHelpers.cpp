//
// Created by Grady Schofield on 12/10/23.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include<MetalHelpers.h>
#include<Exception.h>

static MTL::Device * device = nullptr;

static unordered_map<void const *, MTL::Buffer*> pointerToMetalBuffer;
static unordered_map<MTL::Buffer *, void *> metalBufferToPointer;

namespace MetalHelpers {
    MTL::Device * getDevice() {
        if (!device) {
            device = MTLCreateSystemDefaultDevice();
        }
        return device;
    }

    MTL::Buffer * newBuffer(void * p, long len) {
        MTL::Buffer * ret = device->newBuffer(p, len, MTL::StorageModeShared);
        pointerToMetalBuffer.emplace(p, ret);
        metalBufferToPointer.emplace(ret, p);
        return ret;
    }

    void releaseBuffer(MTL::Buffer * buffer) {
        if (buffer) {
            void *p = mapAt(metalBufferToPointer, buffer);
            pointerToMetalBuffer.erase(p);
            metalBufferToPointer.erase(buffer);
            buffer->release();
        }
    }

    MTL::Buffer * getBuffer(void const * p) {
        return mapAt(pointerToMetalBuffer, p);
    }
}