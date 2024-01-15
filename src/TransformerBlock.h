//
// Created by Grady Schofield on 10/7/23.
//

#ifndef STREAMING_LLAMA_TRANSFORMERBLOCK_H
#define STREAMING_LLAMA_TRANSFORMERBLOCK_H

#include<unistd.h>
#include<sys/mman.h>
#include<fcntl.h>

#include<Checker.h>
#include<Common.h>
#include<LayerNormalization.h>
#include<Matmul.h>
#include<Scratch.h>
#include<TransformerBlockScratch.h>
#include<Weights.h>

using namespace std;

template<typename T>
class TransformerBlock {
    unique_ptr<Weights<T>> queryWeights;
    unique_ptr<Weights<T>> keyWeights;
    unique_ptr<Weights<T>> valueWeights;
    unique_ptr<Weights<T>> attentionNormWeights;
    unique_ptr<Weights<T>> outputWeights;
    unique_ptr<Weights<T>> ffnNormWeights;
    unique_ptr<Weights<T>> ffnWeights1;
    unique_ptr<Weights<T>> ffnWeights2;
    unique_ptr<Weights<T>> ffnWeights3;
    off_t mapOffset;
    size_t mapLength;
    int tensorFile;
    void * mapAddress = nullptr;
    T normEps;
    int currentToken = 0;
    T * ropeFreqs;
    int numHeads;
    int layerIdx;
    bool unmapWeights;
    shared_ptr<Checker> checker;

public:
    TransformerBlock(int layer,
                     map<string, TensorFileInfo> const & tensorFileInfo,
                     int tensorFile,
                     T normEps,
                     T * ropeFreqs,
                     int numHeads,
                     bool unmapWeights = true,
                     shared_ptr<Checker> checker = nullptr)
            : tensorFile(tensorFile),
              normEps(normEps),
              ropeFreqs(ropeFreqs),
              numHeads(numHeads),
              layerIdx(layer),
              unmapWeights(unmapWeights),
              checker(checker)
    {
        vector<pair<string, TensorFileInfo>> layerInfos = getTensorsForLayer(layer, tensorFileInfo);
        mapOffset = layerInfos.front().second.offset;
        TensorFileInfo const & tfi = layerInfos.back().second;
        mapLength = tfi.offset + tfi.numColumns * tfi.leadingDimension * sizeof(T) - mapOffset;
        map<string, TensorFileInfo> layerInfoMap;
        stringstream prefix;
        prefix << "layers." << layer << ".";
        int prefixLen = prefix.str().length();
        for(auto & p : layerInfos) {
            string const & layerName = p.first;
            layerInfoMap.emplace(layerName.substr(prefixLen), p.second);
        }
        mmap();
        queryWeights = make_unique<Weights<T>>(mapAddress, mapOffset, layerInfoMap.at("attention.wq.weight"));
        keyWeights = make_unique<Weights<T>>(mapAddress, mapOffset, layerInfoMap.at("attention.wk.weight"));
        valueWeights = make_unique<Weights<T>>(mapAddress, mapOffset, layerInfoMap.at("attention.wv.weight"));
        attentionNormWeights = make_unique<Weights<T>>(mapAddress, mapOffset, layerInfoMap.at("attention_norm.weight"));
        outputWeights = make_unique<Weights<T>>(mapAddress, mapOffset, layerInfoMap.at("attention.wo.weight"));
        ffnNormWeights = make_unique<Weights<T>>(mapAddress, mapOffset, layerInfoMap.at("ffn_norm.weight"));
        ffnWeights1 = make_unique<Weights<T>>(mapAddress, mapOffset, layerInfoMap.at("feed_forward.w1.weight"));
        ffnWeights2 = make_unique<Weights<T>>(mapAddress, mapOffset, layerInfoMap.at("feed_forward.w2.weight"));
        ffnWeights3 = make_unique<Weights<T>>(mapAddress, mapOffset, layerInfoMap.at("feed_forward.w3.weight"));
        munmap();
    }

    ~TransformerBlock() {
        if(mapAddress) {
            ::munmap(mapAddress, mapLength);
            mapAddress = nullptr;
        }
    }

    void mmap() {
        if (mapAddress) return;
        mapAddress = ::mmap(nullptr, mapLength, PROT_READ, MAP_SHARED, tensorFile, mapOffset);
    }

    void munmap() {
        ::munmap(mapAddress, mapLength);
        mapAddress = nullptr;
    }


    Scratch<T> * evaluate(Scratch<T> * in,
                          int seqlen,
                          shared_ptr<TransformerBlockScratch<T>> transformerBlockScratch,
                          EvaluationTimings & timings) {
        if (layerIdx == 0) {
            fout << "Num tokens so far: " << currentToken << endl;
        }
        timings.start("Transformer input layer norm");
        Scratch<T> * inputCopy = transformerBlockScratch->getInputCopyBuffer();
        T* inPtr = in->getPtr();
        memcpy(inputCopy->getPtr(), inPtr, seqlen * in->getLeadingDimension() * sizeof(T));

        //Layer normalization
#if 1
        LayerNormalization<T>::exec(attentionNormWeights->getMetalBuffer(),
                                       inputCopy->getMetalBuffer(),
                                       queryWeights->getNumColumns(),
                                       inputCopy->getLeadingDimension(),
                                       seqlen,
                                       normEps);
#else
        LayerNormalization<T>::exec(attentionNormWeights->getPtr(),
                                    inputCopy->getPtr(),
                                    queryWeights->getNumColumns(),
                                    inputCopy->getLeadingDimension(),
                                    seqlen,
                                    normEps);
#endif
        timings.finish("Transformer input layer norm");
        if(checker) {
            checker->submitResult(createDataAccessor(inputCopy->getPtr(),
                                                     {queryWeights->getNumColumns(),
                                                      seqlen},
                                                     inputCopy->getLeadingDimension()));
        }
        //M = queryWeights.numRows, K = queryWeights.numCols or embeddingDimension, N = seqlen
        timings.start("Transformer Q*embedding matmul");
        T* wqOutPtr = transformerBlockScratch->getWQout()->getPtr();
        int wqOutLeadingDim = transformerBlockScratch->getWQout()->getLeadingDimension();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            queryWeights->getNumRows(), seqlen, queryWeights->getNumColumns(),
                            1.0,
                            queryWeights->getMetalBuffer(),
                            queryWeights->getLeadingDimension(),
                            inputCopy->getMetalBuffer(),
                            inputCopy->getLeadingDimension(),
                            0.0,
                            transformerBlockScratch->getWQout()->getMetalBuffer(),
                            wqOutLeadingDim);
        timings.finish("Transformer Q*embedding matmul");

        timings.start("Transformer K*embedding matmul");
        T* wkOutPtr = transformerBlockScratch->getWKout(layerIdx)->getPtr();
        int wkOutLeadingDim = transformerBlockScratch->getWKout(layerIdx)->getLeadingDimension();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            keyWeights->getNumRows(), seqlen, keyWeights->getNumColumns(),
                            1.0,
                            keyWeights->getMetalBuffer(),
                            keyWeights->getLeadingDimension(),
                            inputCopy->getMetalBuffer(),
                            inputCopy->getLeadingDimension(),
                            0.0,
                            transformerBlockScratch->getWKout(layerIdx)->getMetalBuffer(),
                            currentToken * wkOutLeadingDim,
                            wkOutLeadingDim);
        timings.finish("Transformer K*embedding matmul");

        timings.start("Transformer V*embedding matmul");
        T * wvOutPtr = transformerBlockScratch->getWVout(layerIdx)->getPtr();
        int wvOutLeadingDim = transformerBlockScratch->getWVout(layerIdx)->getLeadingDimension();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            valueWeights->getNumRows(), seqlen, valueWeights->getNumColumns(),
                            1.0,
                            valueWeights->getMetalBuffer(),
                            valueWeights->getLeadingDimension(),
                            inputCopy->getMetalBuffer(),
                            inputCopy->getLeadingDimension(),
                            0.0,
                            transformerBlockScratch->getWVout(layerIdx)->getMetalBuffer(),
                            currentToken * wvOutLeadingDim,
                            wvOutLeadingDim);
        timings.finish("Transformer V*embedding matmul");

        timings.start("Waiting on first 3 matmuls");
        Metal::waitUntilCompleted(0, reclaimMatvecBuffers);
        timings.finish("Waiting on first 3 matmuls");

        if(checker) {
            checker->submitResult(createDataAccessor(wqOutPtr,
                                                     {queryWeights->getNumRows(),
                                                      seqlen},
                                                     wqOutLeadingDim));
            checker->submitResult(createDataAccessor(wqOutPtr,
                                                     {queryWeights->getNumRows(),
                                                      seqlen},
                                                     wqOutLeadingDim));
            checker->submitResult(createDataAccessor(&wkOutPtr[currentToken * wkOutLeadingDim],
                                                     {keyWeights->getNumRows(),
                                                      seqlen},
                                                     wkOutLeadingDim));
            checker->submitResult(createDataAccessor(&wvOutPtr[currentToken * wvOutLeadingDim],
                                                     {valueWeights->getNumRows(),
                                                      seqlen},
                                                     wvOutLeadingDim));
        }

        // Apply rotary embedding
        int headDimension = queryWeights->getNumRows() / numHeads;
        auto rotaryPositionEmbedding = [this, headDimension, seqlen](T* basePtr, int leadingDimension) {
            for (int j = 0; j < seqlen; ++j) {
                T * ptr = &basePtr[j*leadingDimension];
                int position = currentToken + j;
                for (int head = 0; head < numHeads; ++head) {
                    int k = 0;
                    for (int i = head * headDimension; i < (head + 1) * headDimension; i += 2) {
                        float re = ptr[i];
                        float im = ptr[i+1];
                        float theta = ropeFreqs[k++];
                        float c = cos(position*theta);
                        float s = sin(position*theta);
                        ptr[i] = re * c - im * s;
                        ptr[i + 1] = re * s + im  * c;
                    }
                }
            }
        };
        timings.start("Rotary embeddings on W*Q and W*K");
        rotaryPositionEmbedding(wqOutPtr, wqOutLeadingDim);
        rotaryPositionEmbedding(&wkOutPtr[currentToken * wkOutLeadingDim], wkOutLeadingDim);
        timings.finish("Rotary embeddings on W*Q and W*K");
        if(checker) {
            checker->submitResult(createDataAccessor(wqOutPtr,
                                                     {queryWeights->getNumRows(),
                                                      seqlen},
                                                     wqOutLeadingDim));
            checker->submitResult(createDataAccessor(&wkOutPtr[currentToken * wkOutLeadingDim],
                                                     {keyWeights->getNumRows(),
                                                      seqlen},
                                                     wkOutLeadingDim));
        }

        /*
         * Compute K^T * Q for each head of the attention mechanism
         * We are stepping through horizontal bands of each of K, Q and the output matrix.
         * We are asking for a transpose on a horizontal band of K, not K itself.
         * Imagine the output matrix as numHeads vertically stacked blocks of (cacheSize + seqlen) x seqlen
         */
        Scratch<T> * qkOut = transformerBlockScratch->getQKout();
        T * qkOutPtr = qkOut->getPtr();
        int qkOutLeadingDim = qkOut->getLeadingDimension();
        for(int head = 0; head < numHeads; ++head) {
            int M = currentToken + seqlen;
            int N = seqlen;
            int K = headDimension;
            int inputHeadOffset = head * headDimension;
            int outputHeadOffset = head * (currentToken + seqlen);
            timings.start("Key/Query matrix product");
            multiplyMatrices<T>(CblasColMajor, CblasTrans, CblasNoTrans,
                                M, N, K,
                                1.0,
                                &wkOutPtr[inputHeadOffset],
                                keyWeights->getLeadingDimension(),
                                &wqOutPtr[inputHeadOffset],
                                queryWeights->getLeadingDimension(),
                                0.0,
                                &qkOutPtr[outputHeadOffset],
                                qkOutLeadingDim);
            timings.finish("Key/Query matrix product");
            if (checker) {
                checker->submitResult(createDataAccessor(&qkOutPtr[outputHeadOffset],
                                                         {(currentToken + seqlen),
                                                          seqlen},
                                                         qkOutLeadingDim));
            }
        }
        for(int head = 0; head < numHeads; ++head) {
            int outputHeadOffset = head * (currentToken + seqlen);
            //Compute the softmax with masking
            timings.start("Key/Query masking");
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
            timings.finish("Key/Query masking");
        }
        if(checker) {
            checker->submitResult(createDataAccessor(qkOutPtr,
                                                     {numHeads * (currentToken + seqlen),
                                                      seqlen},
                                                     qkOutLeadingDim));
        }

        timings.start("Values * Masked(Key*Query)");
        Scratch<T> * vqkOut = transformerBlockScratch->getVQKout();
        T * vqkOutPtr = vqkOut->getPtr();
        int vqkOutLeadingDim = vqkOut->getLeadingDimension();
        // Compute wV * softmax(K^T * Q).  The results of each head are "concatenated" with no extra work
        for(int head = 0; head < numHeads; ++head) {
            int headOffset = head * headDimension;
            int qkHeadOffset = head * (currentToken + seqlen);
            int M = headDimension;
            int N = seqlen;
            int K = currentToken + seqlen;
            multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                   M, N, K,
                                   1.0,
                                   &wvOutPtr[headOffset],
                                   wvOutLeadingDim,
                                   &qkOutPtr[qkHeadOffset],
                                   qkOutLeadingDim,
                                   0.0,
                                   &vqkOutPtr[headOffset],
                                   vqkOutLeadingDim);
        }
        timings.finish("Values * Masked(Key*Query)");

        timings.start("Output weights * V*Q*K");
        T* woOutPtr = transformerBlockScratch->getWOout()->getPtr();
        int woOutLeadingDim = transformerBlockScratch->getWOout()->getLeadingDimension();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            outputWeights->getNumRows(), seqlen, outputWeights->getNumColumns(),
                            1.0,
                            outputWeights->getMetalBuffer(),
                            outputWeights->getLeadingDimension(),
                            vqkOut->getMetalBuffer(),
                            vqkOutLeadingDim,
                            0.0,
                            transformerBlockScratch->getWOout()->getMetalBuffer(),
                            woOutLeadingDim);
        Metal::waitUntilCompleted(0, reclaimMatvecBuffers);
        timings.finish("Output weights * V*Q*K");

        timings.start("First residual connection");
        // Handle first residual connection
        Scratch<T> * woOutCopy = transformerBlockScratch->getWOoutCopy();
        T * woOutCopyPtr = woOutCopy->getPtr();
        int inLeadingDim = in->getLeadingDimension();
        for(int j = 0; j < seqlen; ++j) {
            for(int i = 0; i < outputWeights->getNumRows(); ++i) {
                woOutPtr[i + j*woOutLeadingDim] += inPtr[i + j*inLeadingDim];
                woOutCopyPtr[i + j*woOutLeadingDim] = woOutPtr[i + j*woOutLeadingDim];
            }
        }
        timings.finish("First residual connection");

        //FFN layer normalizatoin
        timings.start("FFN layer normalization");
        LayerNormalization<T>::exec(ffnNormWeights->getPtr(),
                                       woOutPtr,
                                       outputWeights->getNumRows(),
                                       woOutLeadingDim,
                                       seqlen,
                                       normEps);
        timings.finish("FFN layer normalization");

        timings.start("W1*out of FFN");
        Scratch<T> * w1Out = transformerBlockScratch->getW1Out();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            ffnWeights1->getNumRows(), seqlen, ffnWeights1->getNumColumns(),
                            1.0,
                            ffnWeights1->getMetalBuffer(),
                            ffnWeights1->getLeadingDimension(),
                            transformerBlockScratch->getWOout()->getMetalBuffer(),
                            woOutLeadingDim,
                            0.0,
                            w1Out->getMetalBuffer(),
                            w1Out->getLeadingDimension());
        timings.finish("W1*out of FFN");

        timings.start("W3*out of FFN");
        Scratch<T> * w3Out = transformerBlockScratch->getW3Out();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            ffnWeights3->getNumRows(), seqlen, ffnWeights3->getNumColumns(),
                            1.0,
                            ffnWeights3->getMetalBuffer(),
                            ffnWeights3->getLeadingDimension(),
                            transformerBlockScratch->getWOout()->getMetalBuffer(),
                            woOutLeadingDim,
                            0.0,
                            w3Out->getMetalBuffer(),
                            w3Out->getLeadingDimension());
        timings.finish("W3*out of FFN");

        timings.start("Wait on first 2 layers of FFN");
        Metal::waitUntilCompleted(0, reclaimMatvecBuffers);
        timings.finish("Wait on first 2 layers of FFN");

        timings.start("silu activation of FFN");
        for(int j = 0; j < seqlen; ++j) {
            T * ptr1 = &w1Out->getPtr()[j * w1Out->getLeadingDimension()];
            T * ptr3 = &w3Out->getPtr()[j * w3Out->getLeadingDimension()];
            for(int i = 0; i < ffnWeights1->getNumRows(); ++i) {
                ptr1[i] = ptr3[i] * ptr1[i] / T(1 + exp(-ptr1[i])); //silu activation on ptr1
            }
        }
        timings.finish("silu activation of FFN");

        timings.start("W2 * activation of FFN");
        Scratch<T> * w2Out = transformerBlockScratch->getW2Out();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            ffnWeights2->getNumRows(), seqlen, ffnWeights2->getNumColumns(),
                            1.0,
                            ffnWeights2->getMetalBuffer(),
                            ffnWeights2->getLeadingDimension(),
                            w1Out->getMetalBuffer(),
                            w1Out->getLeadingDimension(),
                            0.0,
                            w2Out->getMetalBuffer(),
                            w2Out->getLeadingDimension());
        Metal::waitUntilCompleted(0, reclaimMatvecBuffers);
        timings.finish("W2 * activation of FFN");

        timings.start("Final residual connection");
        Scratch<T> * out = transformerBlockScratch->takeFreeIoPtr();
        int outLeadingDim = out->getLeadingDimension();
        for(int j = 0; j < seqlen; ++j) {
            T * outPtr = &out->getPtr()[j * outLeadingDim];
            T * ptr1 = &w2Out->getPtr()[j * w2Out->getLeadingDimension()];
            T * ptr2 = &woOutCopyPtr[j*woOutLeadingDim];
            for(int i = 0; i < ffnWeights2->getNumRows(); ++i) {
                outPtr[i] = ptr1[i] + ptr2[i];
            }
        }
        timings.finish("Final residual connection");
        /*
         * x t0 = layerNorm(p) * attention_norm_weights
         * x tQ = wQ * t0
         * x tK = wK * t0
         * x apply position embedding
         * x t1 = tQ^T * tK
         * x t1 += mask
         * x t2 = row_wise_softmax(t1 / sqrt(row len))
         * x t3 = t2 * wV
         * x concat heads
         * x t4 = p + wO * t3 <-- "p +" coming from the residual connection
         * x t5 = layerNorm(t4) * ffn_norm_weights
         * x t6 = t4 + w2 *(silu(w1*t5) . w3*t5) <-- "t4 + " coming from the residual connection, here . means element-wise multiplication
         * final output weights
         */
        currentToken += seqlen;
        return out;
    }
};

#endif //STREAMING_LLAMA_TRANSFORMERBLOCK_H
