//
// Created by Grady Schofield on 10/1/23.
//

#ifndef STREAMING_LLAMA_TRANSFORMERBLOCKSCRATCH_H
#define STREAMING_LLAMA_TRANSFORMERBLOCKSCRATCH_H

#include<cstdlib>
#include<functional>
#include<iostream>

#include<Common.h>
#include<Scratch.h>

using namespace Common;
using namespace std;

template<typename T>
void allocateScratch(size_t &totalAlloc, void ** p, int alignment, size_t size);

template<typename T>
class TransformerBlockScratch {
    int freeIo = 0;
    Scratch<T> ioPtr[2];
    Scratch<T> inputCopyBuffer;
    Scratch<T> wQout;
    vector<Scratch<T>> wKout;
    vector<Scratch<T>> wVout;
    Scratch<T> wOout;
    Scratch<T> wOoutCopy;
    Scratch<T> qkOut;
    Scratch<T> vqkOut;
    Scratch<T> w1Out;
    Scratch<T> w2Out;
    Scratch<T> w3Out;
    Scratch<T> out;

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
        using namespace std::placeholders;
        auto alignedAlloc = bind(allocateScratch<T>, ref(totalAlloc), _1, _2, _3);
        ioPtr[0] = Scratch<T>(alignedAlloc, 64, embeddingLeadingDim, maxSequenceLength);
        ioPtr[1] = Scratch<T>(alignedAlloc, 64, embeddingLeadingDim, maxSequenceLength);
        inputCopyBuffer = Scratch<T>(alignedAlloc, 64, embeddingLeadingDim, maxSequenceLength);

        //TODO The heads within each matrix aren't aligned.  Does it even matter?  Some experimentation is needed.
        wQout = Scratch<T>(alignedAlloc, 64, qLeadingDim, maxSequenceLength);
        for(int i = 0; i < numLayers; ++i) {
            wKout.push_back(Scratch<T>(alignedAlloc, 64, kLeadingDim, cacheSize + maxSequenceLength));
            wVout.push_back(Scratch<T>(alignedAlloc, 64, vLeadingDim, cacheSize + maxSequenceLength));
        }
        wOout = Scratch<T>(alignedAlloc, 64, oLeadingDim, maxSequenceLength);
        wOoutCopy = Scratch<T>(alignedAlloc, 64, oLeadingDim, maxSequenceLength);

        int qkRows = numHeads * (cacheSize + maxSequenceLength);
        int qkLeadingDim = findAlignment(qkRows, 64);
        qkOut = Scratch<T>(alignedAlloc, 64, qkLeadingDim, maxSequenceLength);
        vqkOut = Scratch<T>(alignedAlloc, 64, vLeadingDim , maxSequenceLength);

        w1Out = Scratch<T>(alignedAlloc, 64, w1LeadingDim, maxSequenceLength);
        w2Out = Scratch<T>(alignedAlloc, 64, w2LeadingDim, maxSequenceLength);
        w3Out = Scratch<T>(alignedAlloc, 64, w3LeadingDim, maxSequenceLength);

        int outLeadingDim = findAlignment(vocabularySize, 64);
        out = Scratch<T>(alignedAlloc, 64, outLeadingDim, 1);
        cout << "Allocated " << setprecision(4) << totalAlloc/1E6f << "MB for scratch\n";
    }

    Scratch<T> takeFreeIoPtr() {
        Scratch<T> tmp = ioPtr[freeIo];
        freeIo = (freeIo + 1) % 2;
        return tmp;
    }

    Scratch<T> getInputCopyBuffer() {
        return inputCopyBuffer;
    }

    Scratch<T> getWQout() {
        return wQout;
    }

    Scratch<T> getWKout(int layerIdx) {
        return wKout[layerIdx];
    }

    Scratch<T> getWVout(int layerIdx) {
        return wVout[layerIdx];
    }

    Scratch<T> getQKout() {
        return qkOut;
    }

    Scratch<T> getVQKout() {
        return vqkOut;
    }

    Scratch<T> getWOout() {
        return wOout;
    }

    Scratch<T> getWOoutCopy() {
        return wOoutCopy;
    }

    Scratch<T> getW1Out() {
        return w1Out;
    }

    Scratch<T> getW2Out() {
        return w2Out;
    }

    Scratch<T> getW3Out() {
        return w3Out;
    }

    Scratch<T> getOut() {
        return out;
    }
};

#endif //STREAMING_LLAMA_TRANSFORMERBLOCKSCRATCH_H
