#include<unistd.h>
#include<sys/mman.h>
#include<fcntl.h>

#include<algorithm>
#include<cmath>
#include<fstream>
#include<iostream>
#include<list>
#include<map>
#include<set>
#include<vector>

#ifdef __APPLE__
#include<Accelerate/Accelerate.h>
#else
#include<mkl.h>
#endif

#include<Bf16.h>
#include<Common.h>
#include<Socket.h>
#include<Timer.h>

/*
 * TODO confirm nothing broke in fp32 on mac
 * TODO option to run deparallelizer to output bfloat16
 * TODO copy Bf16 matrices to fp32 on multiplyMatrices call on Macos
 * TODO call sgemm for bfloat16 when on intel
 * TODO code cleanup
 * TODO Cuda impelmentation
 * TODO handle different number of kv heads for largest model
 */

using namespace std;
using namespace Common;

static ostream & logger = cout;

template<typename T>
class Weights {
    int64_t offsetIntoBlock;
    int numRows;
    int numColumns;
    int leadingDimension;

public:
    Weights() {
    }

    Weights(int64_t mapOffset, TensorFileInfo const & tfi)
        :   offsetIntoBlock(tfi.offset - mapOffset),
            numRows(tfi.numRows),
            numColumns(tfi.numColumns),
            leadingDimension(tfi.leadingDimension)
    {
    }

    T * getPtr(void* base) const {
        return (T*)((uint8_t*)base + offsetIntoBlock);
    }

    int getNumRows() const {
        return numRows;
    }

    int getNumColumns() const {
        return numColumns;
    }

    int getLeadingDimension() const {
        return leadingDimension;
    }
};

template<typename T>
class Scratch {
    T * ptr;
    int leadingDimension;
public:
    Scratch(){
    }

    template<typename Allocator>
    Scratch(Allocator && alignedAlloc, int alignmentBytes, int leadingDimension, int numColumns)
        : leadingDimension(leadingDimension)
    {
        alignedAlloc((void**)&ptr, alignmentBytes, leadingDimension * numColumns * sizeof(T));
    }

    T * getPtr() {
        return ptr;
    }

    int getLeadingDimension() {
        return leadingDimension;
    }
};

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
        auto alignedAlloc = [&totalAlloc](void ** p, int alignment, size_t size) {
            totalAlloc += size;
            posix_memalign(p, alignment, size);
        };
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
        logger << "Allocated " << setprecision(4) << totalAlloc/1E6f << "MB for scratch\n";
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

vector<pair<string, TensorFileInfo>> getTensorsForLayer(int layer, map<string, TensorFileInfo> const & tensorFileInfo) {
    stringstream sstr;
    sstr << "layers." << layer;
    string prefix = sstr.str();
    vector<pair<string, TensorFileInfo>> ret;
    for(auto & p : tensorFileInfo) {
        if(p.first.starts_with(prefix)) {
            ret.push_back(p);
        }
    }
    sort(begin(ret), end(ret), [](auto & x, auto & y) {
        return x.second.offset < y.second.offset;
    });
    return ret;
}

template<typename T>
void layerNormalization(T * weights, T* src, int numRows, int leadingDimension, int seqlen, T normEps) {
    for(int j = 0; j < seqlen; ++j) {
        T accum = 0;
        T* ptr = &src[j * leadingDimension];
        for(int i = 0; i < numRows; ++i) {
            accum += ptr[i] * ptr[i];
        }
        T norm = 1.0 / sqrt(accum/numRows + normEps);
        for(int i = 0; i < numRows; ++i) {
            ptr[i] *= norm * weights[i];
        }
    }
}

#ifdef __APPLE__

template<typename T>
void multiplyMatrices(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                 const int K, const T ALPHA, const T * A, const int LDA,
                 const T * B, const int LDB, const T BETA, T * _Nullable C,
                 const int LDC);

template<>
void multiplyMatrices<float>(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                 const int K, const float ALPHA, const float * A, const int LDA,
                 const float * B, const int LDB, const float BETA, float * C,
                 const int LDC) {
    cblas_sgemm(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
}

template<>
void multiplyMatrices<Bf16>(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                 const int K, const Bf16 ALPHA, const Bf16 * A, const int LDA,
                 const Bf16 * B, const int LDB, const Bf16 BETA, Bf16 * C,
                 const int LDC) {
    //cblas_sgemm(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
}
#else
#endif

template<typename T>
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

public:
    TransformerBlock(int layer,
                     map<string, TensorFileInfo> const & tensorFileInfo,
                     int tensorFile,
                     T normEps,
                     T * ropeFreqs,
                     int numHeads)
        : tensorFile(tensorFile), normEps(normEps), ropeFreqs(ropeFreqs), numHeads(numHeads),
            layerIdx(layer)
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
                        shared_ptr<TransformerBlockScratch<T>> transformerBlockScratch) {
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

        // Apply rotary embedding
        int headDimension = queryWeights.getNumRows() / numHeads;
        auto rotaryPositionEmbedding = [this, headDimension, seqlen](T* basePtr, int leadingDimension) {
            for (int j = 0; j < seqlen; ++j) {
                T * ptr = &basePtr[j*leadingDimension];
                int position = currentToken + j;
                for (int head = 0; head < numHeads; ++head) {
                    int k = 0;
                    for (int i = head * headDimension; i < (head + 1) * headDimension; i += 2) {
                        T re = ptr[i];
                        T im = ptr[i+1];
                        T theta = ropeFreqs[k++];
                        T c = cos(position*theta);
                        T s = sin(position*theta);
                        ptr[i] = re * c - im * s;
                        ptr[i + 1] = re * s + im  * c;
                    }
                }
            }
        };
        rotaryPositionEmbedding(wqOutPtr, wqOutLeadingDim);
        rotaryPositionEmbedding(&wkOutPtr[currentToken * wkOutLeadingDim], wkOutLeadingDim);

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
                T dimNormalizer = 1.0 / sqrt(headDimension);
                for (int i = 0; i < currentToken + seqlen; ++i) {
                    qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] *= dimNormalizer;
                }
                T accum = 0;
                //Leave maxArg in float since we don't have the max value of Bf16
                float maxArg = -numeric_limits<float>::max();
                for (int i = 0; i < currentToken + seqlen; ++i) {
                    maxArg = max(maxArg, (float)qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim]);
                }
                for (int i = 0; i < currentToken + seqlen; ++i) {
                    accum += exp(qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] - T(maxArg));
                }
                T normalizer = 1.0 / accum;
                for (int i = 0; i < currentToken + seqlen; ++i) {
                    T term = exp(qkOutPtr[outputHeadOffset + i + j * qkOutLeadingDim] - T(maxArg));
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

int getLayerCount(map<string, TensorFileInfo> const & tensorFileInfo) {
    set<string> layers;
    for(auto & p : tensorFileInfo) {
        if(p.first.starts_with("layers.")) {
            layers.insert(p.first.substr(7, p.first.find('.', 7)-7));
        }
    }
    return layers.size();
}

vector<pair<string, TensorFileInfo>> getNonTransformerBlockTensors(map<string, TensorFileInfo> const & tensorFileInfo) {
    vector<pair<string, TensorFileInfo>> ret;
    for(auto & p : tensorFileInfo) {
        if(!p.first.starts_with("layers.")) {
            ret.push_back(p);
        }
    }
    sort(begin(ret), end(ret), [](auto & x, auto & y) {
        return x.second.offset < y.second.offset;
    });
    return ret;
}

template<typename T>
class NonTransformerWeights {
    Weights<T> tokenEmbeddings;
    Weights<T> ropeFreqs;
    Weights<T> outputNormalizers;
    Weights<T> outputWeights;
    off_t mapOffset;
    size_t mapLength;
    void * mapAddress = nullptr;

public:
    NonTransformerWeights(map<string, TensorFileInfo> const & tensorFileInfo, int tensorFile) {
        vector<pair<string, TensorFileInfo>> tensorInfos = getNonTransformerBlockTensors(tensorFileInfo);
        mapOffset = tensorInfos.front().second.offset;
        TensorFileInfo const & tfi = tensorInfos.back().second;
        mapLength = tfi.offset + tfi.leadingDimension * tfi.numColumns * sizeof(T) - mapOffset;
        mapAddress = mmap(nullptr, mapLength, PROT_READ, MAP_PRIVATE, tensorFile, mapOffset);
        if(mapAddress == MAP_FAILED) {
            logger << "mmap failed: " << strerror(errno) << "\n";
            throw 3;
        }
        map<string, TensorFileInfo> tensorInfoMap;
        for(auto & p : tensorInfos) {
            string const & tensorName = p.first;
            tensorInfoMap.emplace(tensorName, p.second);
        }
        tokenEmbeddings = Weights<T>(mapOffset, tensorInfoMap.at("tok_embeddings.weight"));
        ropeFreqs = Weights<T>(mapOffset, tensorInfoMap.at("rope.freqs"));
        outputNormalizers = Weights<T>(mapOffset, tensorInfoMap.at("norm.weight"));
        outputWeights = Weights<T>(mapOffset, tensorInfoMap.at("output.weight"));
    }

    ~NonTransformerWeights() {
        if(mapAddress) munmap(mapAddress, mapLength);
    }

    T * getRopeFreqPtr() {
        return ropeFreqs.getPtr(mapAddress);
    }

    int getVocabularySize() const {
        return tokenEmbeddings.getNumColumns();
    }

    void getTokenEmbedding(vector<int> const & tokens, T * out) {
        T const * ptr = tokenEmbeddings.getPtr(mapAddress);
        int i = 0;
        for(int tok : tokens) {
            memcpy(&out[i* tokenEmbeddings.getLeadingDimension()],
                   &ptr[tok * tokenEmbeddings.getLeadingDimension()],
                   tokenEmbeddings.getNumRows() * sizeof(T));
            ++i;
        }
    }

    T * getOutputNormalizers() {
        return outputNormalizers.getPtr(mapAddress);
    }

    T * getOutputWeights() const {
        return outputWeights.getPtr(mapAddress);
    }

    int getOutputLeadingDimension() const {
        return outputWeights.getLeadingDimension();
    }

    void applyOutputLayer(Scratch<T> in, Scratch<T> out, int seqlen, float normEps) {
        layerNormalization<T>(outputNormalizers.getPtr(mapAddress),
                              in.getPtr(),
                              outputWeights.getNumColumns(),
                              in.getLeadingDimension(),
                              seqlen,
                              normEps);

        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            outputWeights.getNumRows(), seqlen, outputWeights.getNumColumns(),
                            1.0,
                            outputWeights.getPtr(mapAddress),
                            outputWeights.getLeadingDimension(),
                            in.getPtr(),
                            in.getLeadingDimension(),
                            0.0,
                            out.getPtr(),
                            out.getLeadingDimension());
    }
};

class LLamaModelInterface {
public:
    virtual vector<float> evaluate(vector<int> const & tokens) = 0;
    virtual ~LLamaModelInterface() {}
};

template<typename T>
class LlamaModel : public LLamaModelInterface {
    shared_ptr<NonTransformerWeights<T>> nonTransformerWeights;
    vector<shared_ptr<TransformerBlock<T>>> transformerBlocks;
    int tensorFile;
    shared_ptr<TransformerBlockScratch<T>> transformerBlockScratch;
    int numHeads;
    int numKvHeads;
    T normEps;
    FileStorageFormat fileStorageFormat;

    map<string, TensorFileInfo> readTensorFileInfoTable(ifstream & ifs) {
        int64_t tensorOffsetTablePos = 0;
        ifs.seekg(0);
        ifs.read((char*)&tensorOffsetTablePos, 8);
        ifs.seekg(tensorOffsetTablePos);
        int numTensors;
        ifs.read((char*) &numTensors, 4);
        map<string, TensorFileInfo> ret;
        for(int i = 0; i < numTensors; ++i) {
            int nameLen = 0;
            ifs.read((char*) &nameLen, 4);
            vector<char> nameBuffer(nameLen);
            ifs.read((char*) nameBuffer.data(), nameLen);
            TensorFileInfo tfi;
            ifs.read((char*) &tfi.offset, 8);
            ifs.read((char*) &tfi.numRows, 4);
            ifs.read((char*) &tfi.numColumns, 4);
            ifs.read((char*) &tfi.leadingDimension, 4);
            ret.emplace(string(nameBuffer.data(), nameBuffer.size()), tfi);
        }
        return ret;
    }

    void readParams(ifstream & ifs) {
        float normEpsFloat;
        ifs.seekg(8);
        ifs.read((char*) &numHeads, 4);
        ifs.read((char*) &numKvHeads, 4);
        ifs.read((char*) &normEpsFloat, 4);
        normEps = normEpsFloat;
        logger << "numHeads: " << numHeads << "\n";
        logger << "numKvHeads: " << numKvHeads << "\n";
        logger << "normEps: " << normEps << "\n";
    }

public:
    LlamaModel(string const & tensorFilename,
               int maxSequenceLength,
               int cacheSize) {
        ifstream ifs(tensorFilename, ios::binary);
        ifs.seekg(20);
        uint8_t storageType;
        ifs.read((char*)&storageType, 1);
        fileStorageFormat = intToFileStorageFormat(storageType);
        map<string, TensorFileInfo> tensorFileInfo = readTensorFileInfoTable(ifs);
        readParams(ifs);
        ifs.close();

        int layerCount = getLayerCount(tensorFileInfo);
        transformerBlockScratch = make_shared<TransformerBlockScratch<T>>(
                maxSequenceLength, cacheSize, numHeads,
                tensorFileInfo.at("tok_embeddings.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wq.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wk.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wv.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wo.weight").leadingDimension,
                tensorFileInfo.at("layers.0.feed_forward.w1.weight").leadingDimension,
                tensorFileInfo.at("layers.0.feed_forward.w2.weight").leadingDimension,
                tensorFileInfo.at("layers.0.feed_forward.w3.weight").leadingDimension,
                tensorFileInfo.at("tok_embeddings.weight").numColumns,
                layerCount);
        tensorFile = open(tensorFilename.c_str(), O_RDONLY);
        nonTransformerWeights = make_shared<NonTransformerWeights<T>>(tensorFileInfo, tensorFile);
        for(int i = 0; i < layerCount; ++i) {
            transformerBlocks.push_back(make_shared<TransformerBlock<T>>(i,
                                                                      tensorFileInfo,
                                                                      tensorFile,
                                                                      normEps,
                                                                      nonTransformerWeights->getRopeFreqPtr(),
                                                                      numHeads));
        }
    }

    ~LlamaModel() {
        close(tensorFile);
    }

    vector<float> evaluate(vector<int> const & tokens) override {
        int seqlen = tokens.size();
        Scratch<T> out = transformerBlockScratch->takeFreeIoPtr();
        nonTransformerWeights->getTokenEmbedding(tokens, out.getPtr());
        for(auto & transformerBlock : transformerBlocks) {
            transformerBlock->mmap();
            out = transformerBlock->evaluate(out,
                                             seqlen,
                                             transformerBlockScratch);
            transformerBlock->munmap();
        }
        vector<float> ret(nonTransformerWeights->getVocabularySize());
        Scratch<T> in = transformerBlockScratch->takeFreeIoPtr();
        memcpy(in.getPtr(),
               &out.getPtr()[out.getLeadingDimension() * (seqlen-1)],
               sizeof(T) * out.getLeadingDimension());
        nonTransformerWeights->applyOutputLayer(in,
                                                transformerBlockScratch->getOut(),
                                                1,
                                                normEps);
        T * outPtr = transformerBlockScratch->getOut().getPtr();
        for(int i = 0; i < nonTransformerWeights->getVocabularySize(); ++i) {
            ret[i] = outPtr[i];
        }
        return ret;
    }

};

shared_ptr<LLamaModelInterface> createLlamaModel(string filename, int maxSequenceLength, int cacheSize) {
    ifstream ifs(filename);
    ifs.seekg(20);
    uint8_t type;
    ifs.read((char*)&type, 1);
    ifs.close();
    FileStorageFormat fileStorageFormat = intToFileStorageFormat(type);
    switch(fileStorageFormat) {
        case Common::Bf16Aligned:
            return make_shared<LlamaModel<Bf16>>(filename, maxSequenceLength, cacheSize);
        case Common::Fp32Aligned:
            return make_shared<LlamaModel<float>>(filename, maxSequenceLength, cacheSize);
        case Common::Cuda:
            throw 1;
    }
}


#define MANUAL_TESTING 0

int main(int argc, char ** argv) {
    //logger = ofstream("log");
#if MANUAL_TESTING == 0
    Socket socket;
#endif

#if MANUAL_TESTING==1
    string filename = "llama_model_7.bin";
#else
    string filename = "release/llama_model_7.bin";
    //string filename = "release/llama_model_notran_emb.bin";
#endif
    ifstream ifs(filename, ios::binary);
    if(ifs.fail()) {
        logger << "Could not load llama model.\n";
        return 1;
    }
    int maxSequenceLength = 100; // 19
    int cacheSize = 500; // 0
    shared_ptr<LLamaModelInterface> model = createLlamaModel(filename, maxSequenceLength, cacheSize);
#if MANUAL_TESTING == 1
    vector<float> logits = model->evaluate({1,518,25580,29962,6028,366,2436,263,22172,3186,1824,297,315,1817,29973,29961,29914,25580,29962});
    for(float t : logits) {
        cout << t << " ";
    }
    return 0;
#endif

#if MANUAL_TESTING == 0
    while (true) {
        int numTokens = 0;
        vector<int> tokens;
        try {
            numTokens = socket.getInt();
            tokens = socket.getIntArray(numTokens);
        } catch(exception & e) {
            logger << "Client disconnected" << endl;
            break;
        }
        Timer timer;
        vector<float> logits = model->evaluate(tokens);
        logger << timer.elapsed()/tokens.size() << " sec per token.  sending back " << logits.size() << " logits" << endl;
        timer.start();
        socket.sendFloatArray(logits);
        socket.sendInt(-1);
    }
#endif

    return 0;


    /*
    list<int> tokenIdx{5000};
    Timer timer;
    model.evaluate(tokenIdx);
    logger << timer.elapsed() << " sec for one token\n";
    timer.start();
    int count = 10;
    for(int i = 0; i < count; ++i) {
        list<int> t{rand() % 32000};
        model.evaluate(t);
    }
    logger << timer.elapsed()/count << " sec per token\n";
     */
    return 0;
}