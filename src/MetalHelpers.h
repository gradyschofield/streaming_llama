//
// Created by Grady Schofield on 12/10/23.
//

#ifndef STREAMING_LLAMA_METALHELPERS_H
#define STREAMING_LLAMA_METALHELPERS_H

#include"Metal.hpp"
#include"Bf16.h"

#include<iostream>
#include<string>

using namespace std;

namespace Metal {
    MTL::Device * getDevice();
    MTL::Buffer * newBuffer(long len);
    MTL::Buffer * newBuffer(void * p, long len);
    MTL::Buffer * getBuffer(void const * p);
    MTL::Library * newLibrary(string const & src);
    void startCapture(string filename, bool eraseExisting = true);
    void finishCapture();


    class MetalBuffer {
        MTL::Buffer * buffer = nullptr;
        size_t size = 0;

    public:
        MetalBuffer(size_t size) {
            buffer = Metal::newBuffer(size);
            this->size = size;
        }

        MetalBuffer(void * ptr, size_t size) {
            buffer = Metal::newBuffer(ptr, size);
            this->size = size;
        }

        MTL::Buffer * getMetalBuffer() {
            return buffer;
        }

        ~MetalBuffer() {
            if (buffer) {
                buffer->release();
            }
        }
    };

    class Function {
        MTL::Function * function = nullptr;
        MTL::ComputePipelineState * computePipelineState = nullptr;

    public:
        Function(){}
        Function(Function const &) = delete;
        Function & operator=(Function const &) = delete;
        Function(Function &&f )
            : function(f.function), computePipelineState(f.computePipelineState)
        {
            f.function = nullptr;
            f.computePipelineState = nullptr;
        }

        Function & operator=(Function && f) {
            function = f.function;
            computePipelineState = f.computePipelineState;
            f.function = nullptr;
            f.computePipelineState = nullptr;
            return *this;
        }

        Function(MTL::Function * function, MTL::ComputePipelineState * computePipelineState)
            : function(function), computePipelineState(computePipelineState)
        {
        }

        MTL::ComputePipelineState * getComputePipelineState() const {
            return computePipelineState;
        }

        void release() {
            if (computePipelineState) computePipelineState->release();
            if (function) function->release();
            computePipelineState = nullptr;
            function = nullptr;
        }

        ~Function() {
            release();
        }
    };

    Function getFunction(MTL::Library * library, string name);
    Function * getFunction(string const & src, string name);

    template<typename T>
    inline void __encode_metal_param2(MTL::ComputeCommandEncoder * commandEncoder, T arg, int idx) {
        commandEncoder->setBytes(&arg, sizeof(arg), idx);
    }

    template<>
    inline void __encode_metal_param2(MTL::ComputeCommandEncoder * commandEncoder, MTL::Buffer * arg, int idx) {
        commandEncoder->setBuffer(arg, 0, idx);
    }

    template<>
    inline void __encode_metal_param2(MTL::ComputeCommandEncoder * commandEncoder, Bf16 * arg, int idx) {
        uint16_t x = arg->getInt();
        commandEncoder->setBytes(&x, 2, idx);
    }

    inline void __encode_metal_param(MTL::ComputeCommandEncoder *, int) {}

    template<typename T, typename ... Args>
    inline void __encode_metal_param(MTL::ComputeCommandEncoder * commandEncoder, int idx, T arg, Args ... args) {
        __encode_metal_param2(commandEncoder, arg, idx);
        __encode_metal_param(commandEncoder, idx+1, args...);
    }

    template<typename ... Args>
    void queueCall(MTL::CommandBuffer * commandBuffer,
              Function const & function,
              int ntx, int nty, int ntz,
              int ngx, int ngy, int ngz,
              Args... args) {
        MTL::Size threadsPerGroup(ntx, nty, ntz);
        MTL::Size threadGroups(ngx, ngy, ngz);
        MTL::ComputeCommandEncoder *commandEncoder = commandBuffer->computeCommandEncoder();
        commandEncoder->setComputePipelineState(function.getComputePipelineState());
        __encode_metal_param(commandEncoder, 0, args...);
        commandEncoder->dispatchThreadgroups(threadGroups, threadsPerGroup);
        commandEncoder->endEncoding();
    }

    template<typename ... Args>
    void callNonblock(MTL::CommandBuffer * commandBuffer,
                     Function const & function,
                     int ntx, int nty, int ntz,
                     int ngx, int ngy, int ngz,
                     Args... args) {
        queueCall(commandBuffer, function, ntx, nty, ntz, ngx, ngy, ngz, args...);
        commandBuffer->commit();
    }

    template<typename ... Args>
    void callAndWait(MTL::CommandBuffer * commandBuffer,
              Function const & function,
              int ntx, int nty, int ntz,
              int ngx, int ngy, int ngz,
              Args... args) {
        callNonblock(commandBuffer, function, ntx, nty, ntz, ngx, ngy, ngz, args...);
        commandBuffer->waitUntilCompleted();
    }

    MTL::CommandBuffer * getCommandBuffer(int idx);

    void waitUntilCompleted(int idx);

    template<typename Func>
    void waitUntilCompleted(int idx, Func && cleanup = [](){}) {
        waitUntilCompleted(idx);
        cleanup();
    }

    template<typename ... Args>
    void callAndWait(int commandBufferIdx,
                     Function const & function,
                     int ntx, int nty, int ntz,
                     int ngx, int ngy, int ngz,
                     Args... args) {
        MTL::CommandBuffer * commandBuffer = getCommandBuffer(commandBufferIdx);
        queueCall(commandBuffer, function, ntx, nty, ntz, ngx, ngy, ngz, args...);
        waitUntilCompleted(commandBufferIdx);
    }
}

#endif //STREAMING_LLAMA_METALHELPERS_H
