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
#include<Matmul.h>
#include<Scratch.h>
#include<TransformerBlockScratch.h>
#include<Weights.h>

using namespace std;

template<typename T, Processor P>
class TransformerBlock {
    Weights<T> queryWeights;
    Weights<T> keyWeights;
    Weights<T> valueWeights;
    Weights<T> attentionNormWeights;
    Weights<T> outputWeights;
    Weights<T> ffnNormWeights;
    Weights<T> ffnWeights1;
    Weights<T> ffnWeights2;
    Weights<T> ffnWeights3;
    off_t mapOffset;
    size_t mapLength;
    int tensorFile;
    void * mapAddress = nullptr;
    T normEps;
    int currentToken = 0;
    T * ropeFreqs;
    int numHeads;
    int layerIdx;
    shared_ptr<Checker> checker;

public:
    TransformerBlock(int layer,
                     map<string, TensorFileInfo> const & tensorFileInfo,
                     int tensorFile,
                     T normEps,
                     T * ropeFreqs,
                     int numHeads,
                     shared_ptr<Checker> checker = nullptr)
            : tensorFile(tensorFile), normEps(normEps), ropeFreqs(ropeFreqs), numHeads(numHeads),
              layerIdx(layer), checker(checker)
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
        queryWeights = Weights<T>(mapOffset, layerInfoMap.at("attention.wq.weight"));
        keyWeights = Weights<T>(mapOffset, layerInfoMap.at("attention.wk.weight"));
        valueWeights = Weights<T>(mapOffset, layerInfoMap.at("attention.wv.weight"));
        attentionNormWeights = Weights<T>(mapOffset, layerInfoMap.at("attention_norm.weight"));
        outputWeights = Weights<T>(mapOffset, layerInfoMap.at("attention.wo.weight"));
        ffnNormWeights = Weights<T>(mapOffset, layerInfoMap.at("ffn_norm.weight"));
        ffnWeights1 = Weights<T>(mapOffset, layerInfoMap.at("feed_forward.w1.weight"));
        ffnWeights2 = Weights<T>(mapOffset, layerInfoMap.at("feed_forward.w2.weight"));
        ffnWeights3 = Weights<T>(mapOffset, layerInfoMap.at("feed_forward.w3.weight"));
    }

    ~TransformerBlock() {
        if(mapAddress) ::munmap(mapAddress, mapLength);
    }

    void mmap() {
        if(mapAddress) return;
        mapAddress = ::mmap(nullptr, mapLength, PROT_READ, MAP_SHARED, tensorFile, mapOffset);
    }

    void munmap() {
        ::munmap(mapAddress, mapLength);
        mapAddress = nullptr;
    }


    Scratch<T> evaluate(Scratch<T> in,
                        int seqlen,
                        shared_ptr<TransformerBlockScratch<T, P>> transformerBlockScratch) {
        Scratch<T> inputCopy = transformerBlockScratch->getInputCopyBuffer();
        T* inPtr = in.getPtr();
        memcpy(inputCopy.getPtr(), inPtr, seqlen * in.getLeadingDimension() * sizeof(T));

        //Layer normalization
        layerNormalization<T>(attentionNormWeights.getPtr(mapAddress),
                              inputCopy.getPtr(),
                              queryWeights.getNumColumns(),
                              inputCopy.getLeadingDimension(),
                              seqlen,
                              normEps);
        if(checker) {
            checker->submitResult(createDataAccessor(inputCopy.getPtr(),
                                                     {queryWeights.getNumColumns(),
                                                      seqlen},
                                                     inputCopy.getLeadingDimension()));
        }
        //M = queryWeights.numRows, K = queryWeights.numCols or embeddingDimension, N = seqlen
        T* wqOutPtr = transformerBlockScratch->getWQout().getPtr();
        int wqOutLeadingDim = transformerBlockScratch->getWQout().getLeadingDimension();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            queryWeights.getNumRows(), seqlen, queryWeights.getNumColumns(),
                            1.0,
                            queryWeights.getPtr(mapAddress),
                            queryWeights.getLeadingDimension(),
                            inputCopy.getPtr(),
                            inputCopy.getLeadingDimension(),
                            0.0,
                            wqOutPtr,
                            wqOutLeadingDim);

        T* wkOutPtr = transformerBlockScratch->getWKout(layerIdx).getPtr();
        int wkOutLeadingDim = transformerBlockScratch->getWKout(layerIdx).getLeadingDimension();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            keyWeights.getNumRows(), seqlen, keyWeights.getNumColumns(),
                            1.0,
                            keyWeights.getPtr(mapAddress),
                            keyWeights.getLeadingDimension(),
                            inputCopy.getPtr(),
                            inputCopy.getLeadingDimension(),
                            0.0,
                            &wkOutPtr[currentToken * wkOutLeadingDim],
                            wkOutLeadingDim);

        T * wvOutPtr = transformerBlockScratch->getWVout(layerIdx).getPtr();
        int wvOutLeadingDim = transformerBlockScratch->getWVout(layerIdx).getLeadingDimension();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            valueWeights.getNumRows(), seqlen, valueWeights.getNumColumns(),
                            1.0,
                            valueWeights.getPtr(mapAddress),
                            valueWeights.getLeadingDimension(),
                            inputCopy.getPtr(),
                            inputCopy.getLeadingDimension(),
                            0.0,
                            &wvOutPtr[currentToken * wvOutLeadingDim],
                            wvOutLeadingDim);

        if(checker) {
            checker->submitResult(createDataAccessor(wqOutPtr,
                                                     {queryWeights.getNumRows(),
                                                      seqlen},
                                                     wqOutLeadingDim));
            checker->submitResult(createDataAccessor(wqOutPtr,
                                                     {queryWeights.getNumRows(),
                                                      seqlen},
                                                     wqOutLeadingDim));
            checker->submitResult(createDataAccessor(&wkOutPtr[currentToken * wkOutLeadingDim],
                                                     {keyWeights.getNumRows(),
                                                      seqlen},
                                                     wkOutLeadingDim));
            checker->submitResult(createDataAccessor(&wvOutPtr[currentToken * wvOutLeadingDim],
                                                     {valueWeights.getNumRows(),
                                                      seqlen},
                                                     wvOutLeadingDim));
        }

        // Apply rotary embedding
        int headDimension = queryWeights.getNumRows() / numHeads;
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
        rotaryPositionEmbedding(wqOutPtr, wqOutLeadingDim);
        rotaryPositionEmbedding(&wkOutPtr[currentToken * wkOutLeadingDim], wkOutLeadingDim);
        if(checker) {
            checker->submitResult(createDataAccessor(wqOutPtr,
                                                     {queryWeights.getNumRows(),
                                                      seqlen},
                                                     wqOutLeadingDim));
            checker->submitResult(createDataAccessor(&wkOutPtr[currentToken * wkOutLeadingDim],
                                                     {keyWeights.getNumRows(),
                                                      seqlen},
                                                     wkOutLeadingDim));
        }

        /*
         * Compute K^T * Q for each head of the attention mechanism
         * We are stepping through horizontal bands of each of K, Q and the output matrix.
         * We are asking for a transpose on a horizontal band of K, not K itself.
         * Imagine the output matrix as numHeads vertically stacked blocks of (cacheSize + seqlen) x seqlen
         */
        Scratch<T> qkOut = transformerBlockScratch->getQKout();
        T * qkOutPtr = qkOut.getPtr();
        int qkOutLeadingDim = qkOut.getLeadingDimension();
        for(int head = 0; head < numHeads; ++head) {
            int M = currentToken + seqlen;
            int N = seqlen;
            int K = headDimension;
            int inputHeadOffset = head * headDimension;
            int outputHeadOffset = head * (currentToken + seqlen);
            multiplyMatrices<T>(CblasColMajor, CblasTrans, CblasNoTrans,
                                M, N, K,
                                1.0,
                                &wkOutPtr[inputHeadOffset],
                                keyWeights.getLeadingDimension(),
                                &wqOutPtr[inputHeadOffset],
                                queryWeights.getLeadingDimension(),
                                0.0,
                                &qkOutPtr[outputHeadOffset],
                                qkOutLeadingDim);
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

        Scratch<T> vqkOut = transformerBlockScratch->getVQKout();
        T * vqkOutPtr = vqkOut.getPtr();
        int vqkOutLeadingDim = vqkOut.getLeadingDimension();
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

        T* woOutPtr = transformerBlockScratch->getWOout().getPtr();
        int woOutLeadingDim = transformerBlockScratch->getWOout().getLeadingDimension();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            outputWeights.getNumRows(), seqlen, outputWeights.getNumColumns(),
                            1.0,
                            outputWeights.getPtr(mapAddress),
                            outputWeights.getLeadingDimension(),
                            vqkOutPtr,
                            vqkOutLeadingDim,
                            0.0,
                            woOutPtr,
                            woOutLeadingDim);

        // Handle first residual connection
        Scratch<T> woOutCopy = transformerBlockScratch->getWOoutCopy();
        T * woOutCopyPtr = woOutCopy.getPtr();
        int inLeadingDim = in.getLeadingDimension();
        for(int j = 0; j < seqlen; ++j) {
            for(int i = 0; i < outputWeights.getNumRows(); ++i) {
                woOutPtr[i + j*woOutLeadingDim] += inPtr[i + j*inLeadingDim];
                woOutCopyPtr[i + j*woOutLeadingDim] = woOutPtr[i + j*woOutLeadingDim];
            }
        }

        //FFN layer normalizatoin
        layerNormalization<T>(ffnNormWeights.getPtr(mapAddress),
                              woOutPtr,
                              outputWeights.getNumRows(),
                              woOutLeadingDim,
                              seqlen,
                              normEps);

        Scratch<T> w1Out = transformerBlockScratch->getW1Out();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            ffnWeights1.getNumRows(), seqlen, ffnWeights1.getNumColumns(),
                            1.0,
                            ffnWeights1.getPtr(mapAddress),
                            ffnWeights1.getLeadingDimension(),
                            woOutPtr,
                            woOutLeadingDim,
                            0.0,
                            w1Out.getPtr(),
                            w1Out.getLeadingDimension());

        Scratch<T> w3Out = transformerBlockScratch->getW3Out();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            ffnWeights3.getNumRows(), seqlen, ffnWeights3.getNumColumns(),
                            1.0,
                            ffnWeights3.getPtr(mapAddress),
                            ffnWeights3.getLeadingDimension(),
                            woOutPtr,
                            woOutLeadingDim,
                            0.0,
                            w3Out.getPtr(),
                            w3Out.getLeadingDimension());

        for(int j = 0; j < seqlen; ++j) {
            T * ptr1 = &w1Out.getPtr()[j * w1Out.getLeadingDimension()];
            T * ptr3 = &w3Out.getPtr()[j * w3Out.getLeadingDimension()];
            for(int i = 0; i < ffnWeights1.getNumRows(); ++i) {
                ptr1[i] = ptr3[i] * ptr1[i] / T(1 + exp(-ptr1[i])); //silu activation on ptr1
            }
        }

        Scratch<T>  w2Out = transformerBlockScratch->getW2Out();
        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            ffnWeights2.getNumRows(), seqlen, ffnWeights2.getNumColumns(),
                            1.0,
                            ffnWeights2.getPtr(mapAddress),
                            ffnWeights2.getLeadingDimension(),
                            w1Out.getPtr(),
                            w1Out.getLeadingDimension(),
                            0.0,
                            w2Out.getPtr(),
                            w2Out.getLeadingDimension());

        Scratch<T> out = transformerBlockScratch->takeFreeIoPtr();
        int outLeadingDim = out.getLeadingDimension();
        for(int j = 0; j < seqlen; ++j) {
            T * outPtr = &out.getPtr()[j * outLeadingDim];
            T * ptr1 = &w2Out.getPtr()[j * w2Out.getLeadingDimension()];
            T * ptr2 = &woOutCopyPtr[j*woOutLeadingDim];
            for(int i = 0; i < ffnWeights2.getNumRows(); ++i) {
                outPtr[i] = ptr1[i] + ptr2[i];
            }
        }
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