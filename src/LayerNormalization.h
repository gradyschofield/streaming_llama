//
// Created by Grady Schofield on 10/8/23.
//

#ifndef STREAMING_LLAMA_LAYERNORMALIZATION_H
#define STREAMING_LLAMA_LAYERNORMALIZATION_H

#include<Common.h>
#include<Cuda.h>

using namespace Common;

template<typename T, Processor P>
class LayerNormalization {
public:
    static void exec(T *weights, T *src, int numRows, int leadingDimension, int seqlen, T normEps);

};

template<typename T>
class LayerNormalization<T, Cpu> {
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
};

template<typename T>
class LayerNormalization<T, Gpu> {
public:
#ifdef __APPLE__
    static void exec(T *weights, T *src, int numRows, int leadingDimension, int seqlen, T normEps) {
        throw Exception("There is no Gpu implementations of layerNormalization for Apple yet.");
    }
#else
    static void exec(T *weights, T *src, int numRows, int leadingDimension, int seqlen, T normEps) {
        Cuda * cuda = getCuda();
        void * args = {};
        cuda->launchKernel("layerNormBf16", args);
    }
#endif
};

#endif //STREAMING_LLAMA_LAYERNORMALIZATION_H
