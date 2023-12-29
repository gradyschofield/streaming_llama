//
// Created by Grady Schofield on 10/1/23.
//

#ifndef STREAMING_LLAMA_TRANSFORMERBLOCKSCRATCH_H
#define STREAMING_LLAMA_TRANSFORMERBLOCKSCRATCH_H

#include<cstdlib>
#include<functional>
#include<iomanip>
#include<iostream>
#include<memory>

#include<Common.h>
#include<Scratch.h>

using namespace Common;
using namespace std;

template<typename T>
class TransformerBlockScratch {
    int freeIo = 0;
    unique_ptr<Scratch<T>> ioPtr[2];
    unique_ptr<Scratch<T>> inputCopyBuffer;
    unique_ptr<Scratch<T>> wQout;
    vector<unique_ptr<Scratch<T>>> wKout;
    vector<unique_ptr<Scratch<T>>> wVout;
    unique_ptr<Scratch<T>> wOout;
    unique_ptr<Scratch<T>> wOoutCopy;
    unique_ptr<Scratch<T>> qkOut;
    unique_ptr<Scratch<T>> vqkOut;
    unique_ptr<Scratch<T>> w1Out;
    unique_ptr<Scratch<T>> w2Out;
    unique_ptr<Scratch<T>> w3Out;
    unique_ptr<Scratch<T>> out;

public:
    TransformerBlockScratch(int maxSequenceLength,
                            int cacheSize,
                            int numHeads,
                            int embeddingLeadingDim,
                            int qLeadingDim,
                            int kLeadingDim,
                            int vLeadingDim,
                            int oLeadingDim,
                            int w1LeadingDim,
                            int w2LeadingDim,
                            int w3LeadingDim,
                            int vocabularySize,
                            int numLayers) {
        size_t totalAlloc = 0;
        ioPtr[0] = make_unique<Scratch<T>>(embeddingLeadingDim, maxSequenceLength);
        ioPtr[1] = make_unique<Scratch<T>>(embeddingLeadingDim, maxSequenceLength);
        inputCopyBuffer = make_unique<Scratch<T>>(embeddingLeadingDim, maxSequenceLength);

        //TODO The heads within each matrix aren't aligned.  Does it even matter?  Some experimentation is needed.
        wQout = make_unique<Scratch<T>>(qLeadingDim, maxSequenceLength);
        for(int i = 0; i < numLayers; ++i) {
            wKout.push_back(make_unique<Scratch<T>>(kLeadingDim, cacheSize + maxSequenceLength));
            wVout.push_back(make_unique<Scratch<T>>(vLeadingDim, cacheSize + maxSequenceLength));
        }
        wOout = make_unique<Scratch<T>>(oLeadingDim, maxSequenceLength);
        wOoutCopy = make_unique<Scratch<T>>(oLeadingDim, maxSequenceLength);

        int qkRows = numHeads * (cacheSize + maxSequenceLength);
        int qkLeadingDim = findAlignment(qkRows, 64);
        qkOut = make_unique<Scratch<T>>(qkLeadingDim, maxSequenceLength);
        vqkOut = make_unique<Scratch<T>>(vLeadingDim , maxSequenceLength);

        w1Out = make_unique<Scratch<T>>(w1LeadingDim, maxSequenceLength);
        w2Out = make_unique<Scratch<T>>(w2LeadingDim, maxSequenceLength);
        w3Out = make_unique<Scratch<T>>(w3LeadingDim, maxSequenceLength);

        int outLeadingDim = findAlignment(vocabularySize, 64);
        out = make_unique<Scratch<T>>(outLeadingDim, 1);
        cout << "Allocated " << setprecision(4) << totalAlloc/1E6f << "MB for scratch\n";
    }

    Scratch<T> * takeFreeIoPtr() {
        Scratch<T> * tmp = ioPtr[freeIo].get();
        freeIo = (freeIo + 1) % 2;
        return tmp;
    }

    Scratch<T> * getInputCopyBuffer() {
        return inputCopyBuffer.get();
    }

    Scratch<T> * getWQout() {
        return wQout.get();
    }

    Scratch<T> * getWKout(int layerIdx) {
        return wKout[layerIdx].get();
    }

    Scratch<T> * getWVout(int layerIdx) {
        return wVout[layerIdx].get();
    }

    Scratch<T> * getQKout() {
        return qkOut.get();
    }

    Scratch<T> * getVQKout() {
        return vqkOut.get();
    }

    Scratch<T> * getWOout() {
        return wOout.get();
    }

    Scratch<T> * getWOoutCopy() {
        return wOoutCopy.get();
    }

    Scratch<T> * getW1Out() {
        return w1Out.get();
    }

    Scratch<T> * getW2Out() {
        return w2Out.get();
    }

    Scratch<T> * getW3Out() {
        return w3Out.get();
    }

    Scratch<T> * getOut() {
        return out.get();
    }
};

#endif //STREAMING_LLAMA_TRANSFORMERBLOCKSCRATCH_H
