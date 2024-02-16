//
// Created by Grady Schofield on 2/11/24.
//

#ifndef STREAMING_LLAMA_MASKING_H
#define STREAMING_LLAMA_MASKING_H

#include<string>

#include<Scratch.h>
#include<Matmul.h>  // TODO: this is here just to get the multihead matvec cleanup function, fix this

using namespace std;

namespace Masking {

    Metal::Function * maskQkProductFunc = nullptr;

    void initMasking() {
        string maskingSrc = R"(
            #include <metal_stdlib>
            using namespace metal;

            /*
             * This is similar to the CPU code below, but j from that code is always 0 in this function.
             */
            kernel void maskQkProduct(device bfloat * qkOut [[buffer(0)]],
                                        constant int64_t & headDimension [[buffer(1)]],
                                        constant int64_t & numHeads [[buffer(2)]],
                                        constant int64_t & currentToken [[buffer(3)]],
                                        constant int64_t & seqlen [[buffer(4)]],
                                        uint2 threadPos [[thread_position_in_threadgroup]],
                                        uint2 threadDim [[threads_per_threadgroup]],
                                        uint2 groupPos [[threadgroup_position_in_grid]],
                                        uint2 groupDim [[threadgroups_per_grid]] ) {
                int64_t head = groupPos.x;
                int outputHeadOffset = head * (currentToken + seqlen);
                for (int i = currentToken + 1; i < currentToken + seqlen; ++i) {
                    qkOut[outputHeadOffset + i] = -BFLT_MAX;
                }
                bfloat dimNormalizer = static_cast<bfloat>(1.0 / sqrt(static_cast<float>(headDimension)));
                for (int i = 0; i < currentToken + seqlen; ++i) {
                    qkOut[outputHeadOffset + i] *= dimNormalizer;
                }
                float accum = 0;
                float maxArg = -FLT_MAX;
                for (int i = 0; i < currentToken + seqlen; ++i) {
                    maxArg = max(maxArg, static_cast<float>(qkOut[outputHeadOffset + i]));
                }
                for (int i = 0; i < currentToken + seqlen; ++i) {
                    accum += exp(qkOut[outputHeadOffset + i] - maxArg);
                }
                bfloat normalizer = static_cast<bfloat>(1.0 / accum);
                for (int i = 0; i < currentToken + seqlen; ++i) {
                    bfloat term = static_cast<bfloat>(exp(qkOut[outputHeadOffset + i] - maxArg));
                    qkOut[outputHeadOffset + i] = term * normalizer;
                }
            }
        )";
        maskQkProductFunc = Metal::getFunction(maskingSrc, "maskQkProduct");
    }

    template<typename T>
    void maskQkProduct(Scratch<T> * qkOut, long headDimension, long numHeads, long currentToken, long seqlen, bool gpu = false) {
        if (gpu) {
            if (!maskQkProductFunc) {
                initMasking();
            }
            MTL::CommandBuffer * commandBuffer = Metal::getCommandBuffer(0);
            Metal::queueCall(commandBuffer, *maskQkProductFunc,
                             1, 1, 1,
                             numHeads, 1, 1,
                             qkOut->getMetalBuffer(),
                             headDimension, numHeads, currentToken, seqlen);

            Metal::waitUntilCompleted(0, cleanupMultiheadMatvecPass);
        } else {
            T *qkOutPtr = qkOut->getPtr();
            int qkOutLeadingDim = qkOut->getLeadingDimension();
            for (int head = 0; head < numHeads; ++head) {
                int outputHeadOffset = head * (currentToken + seqlen);
                //Compute the softmax with masking
                for (int j = 0; j < seqlen; ++j) {
                    for (int i = currentToken + j + 1; i < currentToken + seqlen; ++i) {
                        qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] = -numeric_limits<float>::infinity();
                    }
                    float dimNormalizer = 1.0 / sqrt(headDimension);
                    for (int i = 0; i < currentToken + seqlen; ++i) {
                        qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] *= dimNormalizer;
                    }
                    float accum = 0;
                    //Leave maxArg in float since we don't have the max value of Bf16
                    float maxArg = -numeric_limits<float>::max();
                    for (int i = 0; i < currentToken + seqlen; ++i) {
                        maxArg = max(maxArg, (float) qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim]);
                    }
                    for (int i = 0; i < currentToken + seqlen; ++i) {
                        accum += exp((float) qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] - maxArg);
                    }
                    float normalizer = 1.0 / accum;
                    for (int i = 0; i < currentToken + seqlen; ++i) {
                        float term = exp((float) qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] - maxArg);
                        qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] = term * normalizer;
                    }
                }
            }
        }
    }
}

#endif //STREAMING_LLAMA_MASKING_H
