//
// Created by Grady Schofield on 2/11/24.
//

#ifndef STREAMING_LLAMA_MASKING_H
#define STREAMING_LLAMA_MASKING_H

#include<string>

#include<Scratch.h>

using namespace std;

namespace Masking {

    string maskingSrc = 
    template<typename T>
    void maskQkProduct(Scratch<T> * qkOut, long headDimension, long numHeads, long currentToken, long seqlen) {
        T * qkOutPtr = qkOut->getPtr();
        int qkOutLeadingDim = qkOut->getLeadingDimension();
        for(int head = 0; head < numHeads; ++head) {
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
                    maxArg = max(maxArg, (float)qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim]);
                }
                for (int i = 0; i < currentToken + seqlen; ++i) {
                    accum += exp((float)qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] - maxArg);
                }
                float normalizer = 1.0 / accum;
                for (int i = 0; i < currentToken + seqlen; ++i) {
                    float term = exp((float)qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] - maxArg);
                    qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] = term * normalizer;
                }
            }
        }
    }
}

#endif //STREAMING_LLAMA_MASKING_H
