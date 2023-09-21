#include<sys/socket.h>
#include<sys/un.h>
#include<iostream>
#include<unistd.h>

#include<algorithm>
#include<fstream>
#include<iostream>
#include<list>
#include<map>
#include<set>
#include<vector>

#ifdef __APPLE__
#include<Accelerate/Accelerate.h>
#else
#endif

#include<Common.h>
#include<Timer.h>

/*
 * TODO setup IPC and python program to communicate with this code to encode/decode strings
 * TODO finish the correctness todos (sign on sine etc.)
 * TODO MKL setup
 * TODO handle differenct number of kv heads for largest model
 * TODO option for converting to fp32 after mapping if it seems like it would make the smallest model run OK on Mac
 * TODO Cuda impelmentation
 */

typedef float FloatType; // TODO can we change this for bfloat16 on MacOS Sanoma

using namespace std;
using namespace Common;

ostream & logger = cout;

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

    template<typename T>
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

class Scratch {
    void * ptr;
    int leadingDimension;
public:
    Scratch(){
    }

    template<typename Allocator>
    Scratch(Allocator && alignedAlloc, int alignmentBytes, int leadingDimension, int numColumns)
        : leadingDimension(leadingDimension)
    {
        alignedAlloc((void**)&ptr, alignmentBytes, leadingDimension * numColumns);
    }

    template<typename T>
    T * getPtr() {
        return (T*)ptr;
    }

    int getLeadingDimension() {
        return leadingDimension;
    }
};

class TransformerBlockScratch {
    int freeIo = 0;
    Scratch ioPtr[2];
    Scratch inputCopyBuffer;
    Scratch wQout;
    Scratch wKout;
    Scratch wVout;
    Scratch wOout;
    Scratch wOoutCopy;
    Scratch qkOut;
    Scratch vqkOut;
    Scratch w1Out;
    Scratch w2Out;
    Scratch w3Out;
    Scratch out;

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
                            int vocabularySize) {
        size_t totalAlloc = 0;
        auto alignedAlloc = [&totalAlloc](void ** p, int alignment, size_t size) {
            totalAlloc += size;
            posix_memalign(p, alignment, size);
        };
        ioPtr[0] = Scratch(alignedAlloc, 64, embeddingLeadingDim, maxSequenceLength);
        ioPtr[1] = Scratch(alignedAlloc, 64, embeddingLeadingDim, maxSequenceLength);
        inputCopyBuffer = Scratch(alignedAlloc, 64, embeddingLeadingDim, maxSequenceLength);

        //TODO The heads within each matrix aren't aligned.  Does it even matter?  Some experimentation is needed.
        wQout = Scratch(alignedAlloc, 64, qLeadingDim, maxSequenceLength);
        wKout = Scratch(alignedAlloc, 64, kLeadingDim , cacheSize + maxSequenceLength);
        wVout = Scratch(alignedAlloc, 64, vLeadingDim , cacheSize + maxSequenceLength);
        wOout = Scratch(alignedAlloc, 64, oLeadingDim, maxSequenceLength);
        wOoutCopy = Scratch(alignedAlloc, 64, oLeadingDim, maxSequenceLength);

        int qkRows = numHeads * (cacheSize + maxSequenceLength);
        int qkLeadingDim = findAlignment(qkRows, 64);
        qkOut = Scratch(alignedAlloc, 64, qkLeadingDim, maxSequenceLength);
        vqkOut = Scratch(alignedAlloc, 64, vLeadingDim , maxSequenceLength);

        w1Out = Scratch(alignedAlloc, 64, w1LeadingDim, maxSequenceLength);
        w2Out = Scratch(alignedAlloc, 64, w2LeadingDim, maxSequenceLength);
        w3Out = Scratch(alignedAlloc, 64, w3LeadingDim, maxSequenceLength);

        int outLeadingDim = findAlignment(vocabularySize, 64);
        out = Scratch(alignedAlloc, 64, outLeadingDim, maxSequenceLength);
        logger << "Allocated " << setprecision(4) << totalAlloc/1E6f << "MB for scratch\n";
    }

    Scratch takeFreeIoPtr() {
        Scratch out = ioPtr[freeIo];
        freeIo = (freeIo + 1) % 2;
        return out;
    }

    Scratch getInputCopyBuffer() {
        return inputCopyBuffer;
    }

    Scratch getWQout() {
        return wQout;
    }

    Scratch getWKout() {
        return wKout;
    }

    Scratch getWVout() {
        return wVout;
    }

    Scratch getQKout() {
        return qkOut;
    }

    Scratch getVQKout() {
        return vqkOut;
    }

    Scratch getWOout() {
        return wOout;
    }

    Scratch getWOoutCopy() {
        return wOoutCopy;
    }

    Scratch getW1Out() {
        return w1Out;
    }

    Scratch getW2Out() {
        return w2Out;
    }

    Scratch getW3Out() {
        return w3Out;
    }

    Scratch getOut() {
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
void layerNormalization(T * weights, T* src, int numRows, int leadingDimension, int seqlen, float normEps) {
    for(int j = 0; j < seqlen; ++j) {
        T accum = 0;
        T* ptr = &src[j * leadingDimension];
        for(int i = 0; i < numRows; ++i) {
            accum += ptr[i] * ptr[i];
        }
        float norm = 1.0 / sqrt(accum + normEps);
        for(int i = 0; i < numRows; ++i) {
            ptr[i] *= norm * weights[i];
        }
    }
}

class TransformerBlock {
    Weights queryWeights;
    Weights keyWeights;
    Weights valueWeights;
    Weights attentionNormWeights;
    Weights outputWeights;
    Weights ffnNormWeights;
    Weights ffnWeights1;
    Weights ffnWeights2;
    Weights ffnWeights3;
    off_t mapOffset;
    size_t mapLength;
    int tensorFile;
    void * mapAddress = nullptr;
    float normEps;
    int currentToken = 0;
    float * ropeFreqs;
    int numHeads;

public:
    TransformerBlock(int layer,
                     map<string, TensorFileInfo> const & tensorFileInfo,
                     int tensorFile,
                     float normEps,
                     float * ropeFreqs,
                     int numHeads,
                     int floatSize)
        : tensorFile(tensorFile), normEps(normEps), ropeFreqs(ropeFreqs), numHeads(numHeads)
    {
        vector<pair<string, TensorFileInfo>> layerInfos = getTensorsForLayer(layer, tensorFileInfo);
        mapOffset = layerInfos.front().second.offset;
        TensorFileInfo const & tfi = layerInfos.back().second;
        mapLength = tfi.offset + tfi.numColumns * tfi.leadingDimension * floatSize - mapOffset;
        map<string, TensorFileInfo> layerInfoMap;
        stringstream prefix;
        prefix << "layers." << layer << ".";
        int prefixLen = prefix.str().length();
        for(auto & p : layerInfos) {
            string const & layerName = p.first;
            layerInfoMap.emplace(layerName.substr(prefixLen), p.second);
        }
        queryWeights = Weights(mapOffset, layerInfoMap.at("attention.wq.weight"));
        keyWeights = Weights(mapOffset, layerInfoMap.at("attention.wk.weight"));
        valueWeights = Weights(mapOffset, layerInfoMap.at("attention.wv.weight"));
        attentionNormWeights = Weights(mapOffset, layerInfoMap.at("attention_norm.weight"));
        outputWeights = Weights(mapOffset, layerInfoMap.at("attention.wo.weight"));
        ffnNormWeights = Weights(mapOffset, layerInfoMap.at("ffn_norm.weight"));
        ffnWeights1 = Weights(mapOffset, layerInfoMap.at("feed_forward.w1.weight"));
        ffnWeights2 = Weights(mapOffset, layerInfoMap.at("feed_forward.w2.weight"));
        ffnWeights3 = Weights(mapOffset, layerInfoMap.at("feed_forward.w3.weight"));
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


    template<typename T>
    Scratch evaluate(Scratch in,
                     int seqlen,
                     shared_ptr<TransformerBlockScratch> transformerBlockScratch) {
        Scratch inputCopy = transformerBlockScratch->getInputCopyBuffer();
        T* inPtr = in.getPtr<T>();
        memcpy(inputCopy.getPtr<T>(), inPtr, seqlen * in.getLeadingDimension() * sizeof(T));

        //Layer normalization
        layerNormalization<T>(attentionNormWeights.getPtr<T>(mapAddress),
                              inputCopy.getPtr<T>(),
                              queryWeights.getNumColumns(),
                              inputCopy.getLeadingDimension(),
                              seqlen,
                              normEps);
        //M = queryWeights.numRows, K = queryWeights.numCols or embeddingDimension, N = seqlen
        T* wqOutPtr = transformerBlockScratch->getWQout().getPtr<T>();
        int wqOutLeadingDim = transformerBlockScratch->getWQout().getLeadingDimension();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    queryWeights.getNumRows(), seqlen, queryWeights.getNumColumns(),
                    1.0,
                    queryWeights.getPtr<T>(mapAddress),
                    queryWeights.getLeadingDimension(),
                    inputCopy.getPtr<T>(),
                    inputCopy.getLeadingDimension(),
                    0.0,
                    wqOutPtr,
                    wqOutLeadingDim);

        T* wkOutPtr = transformerBlockScratch->getWKout().getPtr<T>();
        int wkOutLeadingDim = transformerBlockScratch->getWKout().getLeadingDimension();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    keyWeights.getNumRows(), seqlen, keyWeights.getNumColumns(),
                    1.0,
                    keyWeights.getPtr<T>(mapAddress),
                    keyWeights.getLeadingDimension(),
                    inputCopy.getPtr<T>(),
                    inputCopy.getLeadingDimension(),
                    0.0,
                    &wkOutPtr[currentToken * wkOutLeadingDim],
                    wkOutLeadingDim);

        T * wvOutPtr = transformerBlockScratch->getWVout().getPtr<T>();
        int wvOutLeadingDim = transformerBlockScratch->getWVout().getLeadingDimension();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    valueWeights.getNumRows(), seqlen, valueWeights.getNumColumns(),
                    1.0,
                    valueWeights.getPtr<T>(mapAddress),
                    valueWeights.getLeadingDimension(),
                    inputCopy.getPtr<T>(),
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
        Scratch wQOut = transformerBlockScratch->getWQout();
        rotaryPositionEmbedding(wQOut.getPtr<FloatType>(), wQOut.getLeadingDimension());
        Scratch wKOut = transformerBlockScratch->getWKout();
        rotaryPositionEmbedding(wKOut.getPtr<FloatType>(), wKOut.getLeadingDimension());

        /*
         * Compute K^T * Q for each head of the attention mechanism
         * We are stepping through horizontal bands of each of K, Q and the output matrix.
         * We are asking for a transpose on a horizontal band of K, not K itself.
         * Imagine the output matrix as numHeads vertically stacked blocks of (cacheSize + seqlen) x seqlen
         */
        Scratch qkOut = transformerBlockScratch->getQKout();
        T * qkOutPtr = qkOut.getPtr<T>();
        int qkOutLeadingDim = qkOut.getLeadingDimension();
        for(int head = 0; head < numHeads; ++head) {
            int M = currentToken + seqlen;
            int N = seqlen;
            int K = headDimension;
            int headOffset = head * headDimension;
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        M, N, K,
                        1.0,
                        &wkOutPtr[headOffset],
                        keyWeights.getLeadingDimension(),
                        &wqOutPtr[headOffset],
                        queryWeights.getLeadingDimension(),
                        0.0,
                        &qkOutPtr[headOffset],
                        qkOutLeadingDim);
            //Compute the softmax with masking
            for (int j = 0; j < seqlen; ++j) {
                for (int i = currentToken + 1; i < currentToken + seqlen; ++i) {
                    qkOutPtr[headOffset + i + j * qkOutLeadingDim] = -numeric_limits<float>::infinity();
                }
                //TODO check that its sqrt(headDimension)
                float dimNormalizer = 1.0 / sqrt(headDimension);
                for (int i = currentToken; i < currentToken + seqlen; ++i) {
                    qkOutPtr[headOffset + i + j * qkOutLeadingDim] *= dimNormalizer;
                }
                float accum = 0;
                float maxArg = -numeric_limits<float>::max();
                for (int i = currentToken; i < currentToken + seqlen; ++i) {
                    maxArg = max(maxArg, qkOutPtr[headOffset + i + j * qkOutLeadingDim]);
                }
                for (int i = currentToken; i < currentToken + seqlen; ++i) {
                    accum += exp(qkOutPtr[headOffset + i + j * qkOutLeadingDim] - maxArg);
                }
                float normalizer = 1.0 / accum;
                for (int i = currentToken; i < currentToken + seqlen; ++i) {
                    float term = exp(qkOutPtr[headOffset + i + j * qkOutLeadingDim] - maxArg);
                    qkOutPtr[headOffset + i + j * qkOutLeadingDim] *= term * normalizer;
                }
            }
        }

        Scratch vqkOut = transformerBlockScratch->getVQKout();
        T * vqkOutPtr = vqkOut.getPtr<T>();
        int vqkOutLeadingDim = vqkOut.getLeadingDimension();
        // Compute wV * softmax(K^T * Q).  The results of each head are "concatenated" with no extra work
        for(int head = 0; head < numHeads; ++head) {
            int headOffset = head * headDimension;
            int M = queryWeights.getNumRows();
            int N = seqlen;
            int K = currentToken + seqlen;
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K,
                        1.0,
                        &wvOutPtr[headOffset],
                        wvOutLeadingDim,
                        &qkOutPtr[headOffset],
                        qkOutLeadingDim,
                        0.0,
                        &vqkOutPtr[headOffset],
                        vqkOutLeadingDim);
        }

        T* woOutPtr = transformerBlockScratch->getWOout().getPtr<T>();
        int woOutLeadingDim = transformerBlockScratch->getWOout().getLeadingDimension();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    valueWeights.getNumRows(), seqlen, valueWeights.getNumColumns(),
                    1.0,
                    valueWeights.getPtr<T>(mapAddress),
                    valueWeights.getLeadingDimension(),
                    vqkOutPtr,
                    vqkOutLeadingDim,
                    0.0,
                    woOutPtr,
                    woOutLeadingDim);

        // Handle first residual connection
        Scratch woOutCopy = transformerBlockScratch->getWOoutCopy();
        T * woOutCopyPtr = woOutCopy.getPtr<T>();
        int inLeadingDim = in.getLeadingDimension();
        for(int j = 0; j < seqlen; ++j) {
            for(int i = 0; i < outputWeights.getNumRows(); ++i) {
                woOutPtr[i + j*woOutLeadingDim] += inPtr[i + j*inLeadingDim];
                woOutCopyPtr[i + j*woOutLeadingDim] = woOutPtr[i + j*woOutLeadingDim];
            }
        }

        //FFN layer normalizatoin
        layerNormalization<T>(ffnNormWeights.getPtr<T>(mapAddress),
                              woOutPtr,
                              outputWeights.getNumRows(),
                              woOutLeadingDim,
                              seqlen,
                              normEps);

        Scratch w1Out = transformerBlockScratch->getW1Out();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    ffnWeights1.getNumRows(), seqlen, ffnWeights1.getNumColumns(),
                    1.0,
                    ffnWeights1.getPtr<T>(mapAddress),
                    ffnWeights1.getLeadingDimension(),
                    woOutPtr,
                    woOutLeadingDim,
                    0.0,
                    w1Out.getPtr<T>(),
                    w1Out.getLeadingDimension());

        Scratch w3Out = transformerBlockScratch->getW3Out();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    ffnWeights3.getNumRows(), seqlen, ffnWeights3.getNumColumns(),
                    1.0,
                    ffnWeights3.getPtr<T>(mapAddress),
                    ffnWeights3.getLeadingDimension(),
                    woOutPtr,
                    woOutLeadingDim,
                    0.0,
                    w3Out.getPtr<T>(),
                    w3Out.getLeadingDimension());

        for(int j = 0; j < seqlen; ++j) {
            T * ptr1 = &w1Out.getPtr<T>()[j * w1Out.getLeadingDimension()];
            T * ptr3 = &w3Out.getPtr<T>()[j * w3Out.getLeadingDimension()];
            for(int i = 0; i < ffnWeights1.getNumRows(); ++i) {
                ptr1[i] = ptr3[i] * ptr1[i] / (1 + exp(-ptr1[i])); //silu activation on ptr1
            }
        }

        Scratch w2Out = transformerBlockScratch->getW2Out();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    ffnWeights2.getNumRows(), seqlen, ffnWeights2.getNumColumns(),
                    1.0,
                    ffnWeights2.getPtr<T>(mapAddress),
                    ffnWeights2.getLeadingDimension(),
                    w1Out.getPtr<T>(),
                    w1Out.getLeadingDimension(),
                    0.0,
                    w2Out.getPtr<T>(),
                    w2Out.getLeadingDimension());

        Scratch out = transformerBlockScratch->takeFreeIoPtr();
        int outLeadingDim = out.getLeadingDimension();
        for(int j = 0; j < seqlen; ++j) {
            T * outPtr = &out.getPtr<T>()[j * outLeadingDim];
            T * ptr1 = &w2Out.getPtr<T>()[j * w2Out.getLeadingDimension()];
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

class NonTransformerWeights {
    Weights tokenEmbeddings;
    Weights ropeFreqs;
    Weights outputNormalizers;
    Weights outputWeights;
    off_t mapOffset;
    size_t mapLength;
    void * mapAddress = nullptr;

public:
    NonTransformerWeights(map<string, TensorFileInfo> const & tensorFileInfo, int tensorFile, int floatSize) {
        vector<pair<string, TensorFileInfo>> tensorInfos = getNonTransformerBlockTensors(tensorFileInfo);
        mapOffset = tensorInfos.front().second.offset;
        TensorFileInfo const & tfi = tensorInfos.back().second;
        mapLength = tfi.offset + tfi.leadingDimension * tfi.numColumns * floatSize - mapOffset;
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
        tokenEmbeddings = Weights(mapOffset, tensorInfoMap.at("tok_embeddings.weight"));
        ropeFreqs = Weights(mapOffset, tensorInfoMap.at("rope.freqs"));
        outputNormalizers = Weights(mapOffset, tensorInfoMap.at("norm.weight"));
        outputWeights = Weights(mapOffset, tensorInfoMap.at("output.weight"));
    }

    ~NonTransformerWeights() {
        if(mapAddress) munmap(mapAddress, mapLength);
    }

    template<typename T>
    T * getRopeFreqPtr() {
        return ropeFreqs.getPtr<T>(mapAddress);
    }

    int getTokenEmbeddingDimension() const {
        return tokenEmbeddings.getNumRows();
    }

    void getTokenEmbedding(list<int> const & tokens, FloatType * out) {
        FloatType const * ptr = tokenEmbeddings.getPtr<FloatType>(mapAddress);
        int i = 0;
        for(int tok : tokens) {
            memcpy(&out[i* tokenEmbeddings.getLeadingDimension()],
                   &ptr[tok * tokenEmbeddings.getLeadingDimension()],
                   tokenEmbeddings.getNumRows() * sizeof(FloatType));
            ++i;
        }
    }

    template<typename T>
    T * getOutputNormalizers() {
        return outputNormalizers.getPtr<T>(mapAddress);
    }

    template<typename T>
    T * getOutputWeights() const {
        return outputWeights.getPtr<T>(mapAddress);
    }

    int getOutputLeadingDimension() const {
        return outputWeights.getLeadingDimension();
    }

    template<typename T>
    void applyOutputLayer(Scratch in, Scratch out, int seqlen, float normEps) {
        layerNormalization<T>(outputNormalizers.getPtr<T>(mapAddress),
                              in.getPtr<FloatType>(),
                              outputWeights.getNumColumns(),
                              in.getLeadingDimension(),
                              seqlen,
                              normEps);

        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    outputWeights.getNumRows(), seqlen, outputWeights.getNumColumns(),
                    1.0,
                    outputWeights.getPtr<T>(mapAddress),
                    outputWeights.getLeadingDimension(),
                    in.getPtr<T>(),
                    in.getLeadingDimension(),
                    0.0,
                    out.getPtr<T>(),
                    out.getLeadingDimension());
    }
};

class LlamaModel {
    shared_ptr<NonTransformerWeights> nonTransformerWeights;
    vector<shared_ptr<TransformerBlock>> transformerBlocks;
    int tensorFile;
    shared_ptr<TransformerBlockScratch> transformerBlockScratch;
    float normEps;

public:
    LlamaModel(map<string, TensorFileInfo> const & tensorFileInfo,
               string const & tensorFilename,
               int numHeads,
               int maxSequenceLength,
               int cacheSize,
               float normEps,
               int floatSize)
       : normEps(normEps)
   {
        transformerBlockScratch = make_shared<TransformerBlockScratch>(
                maxSequenceLength, cacheSize, numHeads,
                tensorFileInfo.at("tok_embeddings.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wq.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wk.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wv.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wo.weight").leadingDimension,
                tensorFileInfo.at("layers.0.feed_forward.w1.weight").leadingDimension,
                tensorFileInfo.at("layers.0.feed_forward.w2.weight").leadingDimension,
                tensorFileInfo.at("layers.0.feed_forward.w3.weight").leadingDimension,
                tensorFileInfo.at("tok_embeddings.weight").numColumns);
        tensorFile = open(tensorFilename.c_str(), O_RDONLY);
        int layerCount = getLayerCount(tensorFileInfo);
        nonTransformerWeights = make_shared<NonTransformerWeights>(tensorFileInfo, tensorFile, floatSize);
        for(int i = 0; i < layerCount; ++i) {
            transformerBlocks.push_back(make_shared<TransformerBlock>(i,
                                                                      tensorFileInfo,
                                                                      tensorFile,
                                                                      normEps,
                                                                      nonTransformerWeights->getRopeFreqPtr<float>(),
                                                                      numHeads,
                                                                      floatSize));
        }
    }

    ~LlamaModel() {
        close(tensorFile);
    }

    vector<FloatType> evaluate(list<int> const & tokens) {
        int seqlen = tokens.size();
        vector<FloatType> ret;
        Scratch out = transformerBlockScratch->takeFreeIoPtr();
        nonTransformerWeights->getTokenEmbedding(tokens, out.getPtr<FloatType>());
        int k = 0;
        int blockLimit = 2;
        for(auto & transformerBlock : transformerBlocks) {
            if(k > blockLimit) break;
            transformerBlock->mmap();
            out = transformerBlock->evaluate<FloatType>(out,
                                                        seqlen,
                                                        transformerBlockScratch);
            ++k;
        }
        nonTransformerWeights->applyOutputLayer<FloatType>(out,
                                                           transformerBlockScratch->getOut(),
                                                           seqlen,
                                                           normEps);
        return ret;
    }

};


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

tuple<int, int, float> readParams(ifstream & ifs) {
    ifs.seekg(8);
    int numHeads = 0, numKvHeads = 0;
    float normEps = 0;
    ifs.read((char*) &numHeads, 4);
    ifs.read((char*) &numKvHeads, 4);
    ifs.read((char*) &normEps, 4);
    logger << "numHeads: " << numHeads << "\n";
    logger << "numKvHeads: " << numKvHeads << "\n";
    logger << "normEps: " << normEps << "\n";
    return tie(numHeads, numKvHeads, normEps);
}

string const socketPath = "/tmp/codellama_evaluator.sock";

pair<int, int> setupSockets() {
    int serverSocket;
    struct sockaddr_un serverAddress;

    serverSocket = socket(AF_UNIX, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        logger << "Failed to create socket." << endl;
        return {-1,-1};
    }
    memset(&serverAddress, 0, sizeof(serverAddress));
    serverAddress.sun_family = AF_UNIX;
    strncpy(serverAddress.sun_path, socketPath.c_str(), sizeof(serverAddress.sun_path) - 1);
    unlink(socketPath.c_str());

    if (bind(serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) == -1) {
        logger << "Bind error." << endl;
        return {-1,-1};
    }

    if (listen(serverSocket, 1) == -1) {
        logger << "Listen error." << endl;
        return {-1,-1};
    }

    int clientSocket;
    struct sockaddr_un clientAddress;
    socklen_t clientAddressSize;
    clientAddressSize = sizeof(clientAddress);
    clientSocket = accept(serverSocket, (struct sockaddr *)&clientAddress, &clientAddressSize);
    if (clientSocket == -1) {
        logger << "Accept error." << endl;
        return {-1,-1};
    }
    return make_pair(clientSocket, serverSocket);
}

int main(int argc, char ** argv) {
    //logger = ofstream("log");
    /*

    int clientSocket = 0, serverSocket = 0;
    tie(clientSocket, serverSocket) = setupSockets();
    if(clientSocket == -1) return 1;

    while (true) {
        int buffer[256];
        int num_bytes = recv(clientSocket, buffer, sizeof(buffer), 0);

        if (num_bytes <= 0) {
            logger << "Receive error or client disconnected." << std::endl;
            break;
        }

        for (int i = 0; i < num_bytes / sizeof(int); ++i) {
            buffer[i]++;
        }

        send(clientSocket, buffer, num_bytes, 0);
    }

    close(clientSocket);
    close(serverSocket);
    unlink(socketPath.c_str());
    return 0;

     */

    string filename = "llama_model_7.bin";
    ifstream ifs(filename, ios::binary);
    if(ifs.fail()) {
        logger << "Could not load llama model.\n";
        return 1;
    }
    map<string, TensorFileInfo> tensorFileInfo = readTensorFileInfoTable(ifs);
    int numHeads, numKvHeads;
    float normEps;
    tie(numHeads, numKvHeads, normEps) = readParams(ifs);
    ifs.close();
    int maxSequenceLength = 1;
    int cacheSize = 500;
    int floatSize = 4;
    LlamaModel model(tensorFileInfo, filename, numHeads, maxSequenceLength, cacheSize, normEps, floatSize);
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
    return 0;
}