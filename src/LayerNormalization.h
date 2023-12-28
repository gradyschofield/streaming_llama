//
// Created by Grady Schofield on 10/8/23.
//

#ifndef STREAMING_LLAMA_LAYERNORMALIZATION_H
#define STREAMING_LLAMA_LAYERNORMALIZATION_H

#include<Common.h>
#include<MetalHelpers.h>

using namespace Common;

template<typename T>
class LayerNormalization {
public:
    static void exec(T *weights, T *src, int numRows, int leadingDimension, int seqlen, T normEps) {
        for (int j = 0; j < seqlen; ++j) {
            float accum = 0;
            T *ptr = &src[j * leadingDimension];
            for (int i = 0; i < numRows; ++i) {
                accum += (float) ptr[i] * (float) ptr[i];
            }
            float norm = 1.0 / sqrt(accum / numRows + normEps);
            for (int i = 0; i < numRows; ++i) {
                ptr[i] *= (float) weights[i] * norm;
            }
        }
    }

    static void exec(MTL::Buffer *weights,
                     MTL::Buffer *src,
                     int numRows,
                     int leadingDimension,
                     int seqlen,
                     T normEps) {
        static string kernelSrc = R"(
            #include <metal_stdlib>
            #include <metal_compute>
            using namespace metal;

            kernel void layerNormalization(
                                        device bfloat * src [[buffer(0)]],
                                        device const bfloat * weights [[buffer(1)]],
                                        constant int32_t & numRows [[buffer(2)]],
                                        constant int32_t & leadingDimension [[buffer(3)]],
                                        constant bfloat & normEps [[buffer(4)]],
                                        uint threadPos [[thread_position_in_threadgroup]],
                                        uint groupPos [[threadgroup_position_in_grid]]) {
                device bfloat * v = src + groupPos * leadingDimension;
                threadgroup float partials[32];
                float accum = 0;
                for (int i = threadPos; i < numRows; i += 32) {
                    accum += static_cast<float>(v[i]) * static_cast<float>(v[i]);
                }
                partials[threadPos] = accum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                accum = 0;
                for (int i = 0; i < 32; ++i) {
                    accum += partials[i];
                }
                float norm = 1.0 / sqrt(accum / numRows + static_cast<float>(normEps));
                for (int i = threadPos; i < numRows; i+=32) {
                    v[i] = static_cast<bfloat>(static_cast<float>(weights[i]) * norm);
                }
            }
        )";
        Metal::Function * function = Metal::getFunction(kernelSrc, "layerNormalization");
        Metal::callAndWait(0, *function,
                           32, 1, 1,
                           seqlen, 1, 1,
                           src,
                           weights,
                           numRows,
                           leadingDimension,
                           normEps);
        cout << "Finished layer norm" << endl;
    }
};

#endif //STREAMING_LLAMA_LAYERNORMALIZATION_H
