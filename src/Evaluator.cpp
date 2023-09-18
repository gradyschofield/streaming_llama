#include<fstream>
#include<iostream>
#include<list>
#include<map>
#include<set>

#ifdef __APPLE__
#include<Accelerate/Accelerate.h>
#else
#endif

#include<Common.h>

typedef float FloatType; // TODO can we change this for bfloat16 on MacOS Sanoma

using namespace std;
using namespace Common;

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
        return (T*)((uint8_t*)base - offsetIntoBlock);
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

class TransformerBlockScratch {
    int freeIo = 0;
    void * ioPtr[2];
    void * inputCopyBuffer;
    void * wQout;
    void * wKout;
    void * wVout;
    void * wOout;
    void * wOoutCopy;
    void * qkOut;
    void * vqkOut;
    void * w1Out;
    void * w2Out;
    void * w3Out;

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
                            int w3LeadingDim) {
        size_t totalAlloc = 0;
        auto alignedAlloc = [&totalAlloc](void ** p, int alignment, size_t size) {
            totalAlloc += size;
            posix_memalign(p, alignment, size);
        };
        alignedAlloc((void**)&ioPtr[0], 64, embeddingLeadingDim * maxSequenceLength);
        alignedAlloc((void**)&ioPtr[1], 64, embeddingLeadingDim * maxSequenceLength);
        alignedAlloc((void**)&inputCopyBuffer, 64, embeddingLeadingDim * maxSequenceLength);

        //TODO The heads within each matrix aren't aligned.  Does it even matter?  Some experimentation is needed.
        alignedAlloc((void**)&wQout, 64, qLeadingDim * maxSequenceLength);
        alignedAlloc((void**)&wKout, 64, kLeadingDim * (cacheSize + maxSequenceLength));
        alignedAlloc((void**)&wVout, 64, vLeadingDim * (cacheSize + maxSequenceLength));
        alignedAlloc((void**)&wOout, 64, oLeadingDim * maxSequenceLength);
        alignedAlloc((void**)&wOoutCopy, 64, oLeadingDim * maxSequenceLength);

        int qkRows = numHeads * maxSequenceLength;
        int qkLeadingDim = findAlignment(qkRows, 64);
        alignedAlloc((void**)&qkOut, 64, qkLeadingDim * (cacheSize + maxSequenceLength));
        alignedAlloc((void**)&vqkOut, 64, vLeadingDim * maxSequenceLength);

        alignedAlloc((void**)&w1Out, 64, w1LeadingDim * maxSequenceLength);
        alignedAlloc((void**)&w2Out, 64, w2LeadingDim * maxSequenceLength);
        alignedAlloc((void**)&w3Out, 64, w3LeadingDim * maxSequenceLength);
        cout << "Allocated " << setprecision(4) << totalAlloc/1E6f << "MB for scratch\n";
    }

    template<typename T>
    T * takeFreeIoPtr() {
        void * out = ioPtr[freeIo];
        freeIo = (freeIo + 1) % 2;
        return (T*)out;
    }

    template<typename T>
    T * getInputCopyBuffer() {
        return (T*) inputCopyBuffer;
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

public:
    TransformerBlock(int layer, map<string, TensorFileInfo> const & tensorFileInfo, int tensorFile)
        : tensorFile(tensorFile)
    {
        vector<pair<string, TensorFileInfo>> layerInfos = getTensorsForLayer(layer, tensorFileInfo);
        mapOffset = layerInfos.front().second.offset;
        TensorFileInfo const & tfi = layerInfos.back().second;
        mapLength = tfi.offset + tfi.numColumns * tfi.leadingDimension - mapOffset;
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
        mapAddress = ::mmap(nullptr, mapLength, PROT_READ, MAP_SHARED, tensorFile, mapOffset);
    }

    void munmap() {
        ::munmap(mapAddress, mapLength);
        mapAddress = nullptr;
    }

    template<typename T>
    T * evaluate(T * p, int seqlen, shared_ptr<TransformerBlockScratch> transformerBlockScratch) {
        T * out = transformerBlockScratch->takeFreeIoPtr<T>();
        /*
         * t0 = layerNorm(p) * attention_norm_weights
         * tQ = wQ * t0
         * tK = wK * t0
         * apply position embedding
         * t1 = tQ^T * tK
         * t1 += mask
         * t2 = row_wise_softmax(t1 / sqrt(row len))
         * t3 = t2 * wV
         * concat heads
         * t4 = p + wO * t3 <-- "p +" coming from the residual connection
         * t5 = layerNorm(t4) * ffn_norm_weights
         * t6 = t4 + w2 *(silu(w1*t5) . w3*t5) <-- "t4 + " coming from the residual connection, here . means element-wise multiplication
         */
        /*
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    M, N, K, 1.0, a, K, b, K, 0.0, c, M);
                    */
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
    NonTransformerWeights(map<string, TensorFileInfo> const & tensorFileInfo, int tensorFile) {
        vector<pair<string, TensorFileInfo>> tensorInfos = getNonTransformerBlockTensors(tensorFileInfo);
        mapOffset = tensorInfos.front().second.offset;
        TensorFileInfo const & tfi = tensorInfos.back().second;
        mapLength = tfi.offset + tfi.leadingDimension * tfi.numColumns - mapOffset;
        mapAddress = mmap(nullptr, mapLength, PROT_READ, MAP_PRIVATE, tensorFile, mapOffset);
        if(mapAddress == MAP_FAILED) {
            cout << "mmap failed: " << strerror(errno) << "\n";
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
};

class LlamaModel {
    shared_ptr<NonTransformerWeights> nonTransformerWeights;
    vector<shared_ptr<TransformerBlock>> transformerBlocks;
    int tensorFile;
    shared_ptr<TransformerBlockScratch> transformerBlockScratch;

public:
    LlamaModel(map<string, TensorFileInfo> const & tensorFileInfo,
               string const & tensorFilename,
               int numHeads,
               int maxSequenceLength,
               int cacheSize) {
        transformerBlockScratch = make_shared<TransformerBlockScratch>(
                maxSequenceLength, cacheSize, numHeads,
                tensorFileInfo.at("tok_embeddings.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wq.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wk.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wv.weight").leadingDimension,
                tensorFileInfo.at("layers.0.attention.wo.weight").leadingDimension,
                tensorFileInfo.at("layers.0.feed_forward.w1.weight").leadingDimension,
                tensorFileInfo.at("layers.0.feed_forward.w2.weight").leadingDimension,
                tensorFileInfo.at("layers.0.feed_forward.w3.weight").leadingDimension);
        tensorFile = open(tensorFilename.c_str(), O_RDONLY);
        int layerCount = getLayerCount(tensorFileInfo);
        nonTransformerWeights = make_shared<NonTransformerWeights>(tensorFileInfo, tensorFile);
        for(int i = 0; i < layerCount; ++i) {
            transformerBlocks.push_back(make_shared<TransformerBlock>(i, tensorFileInfo, tensorFile));
        }
    }

    ~LlamaModel() {
        close(tensorFile);
    }

    vector<FloatType> evaluate(list<int> const & tokens) {
        int seqlen = tokens.size();
        vector<FloatType> ret;
        FloatType * out = transformerBlockScratch->takeFreeIoPtr<FloatType>();
        nonTransformerWeights->getTokenEmbedding(tokens, out);
        for(auto & transformerBlock : transformerBlocks) {
            out = transformerBlock->evaluate(out, seqlen, transformerBlockScratch);
        }
        //TODO Do layer normalization and multiplication of output weights
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
        ret.emplace(std::move(nameBuffer), tfi);
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
    cout << "numHeads: " << numHeads << "\n";
    cout << "numKvHeads: " << numKvHeads << "\n";
    cout << "normEps: " << normEps << "\n";
    return tie(numHeads, numKvHeads, normEps);
}

int main(int argc, char ** argv) {
    string filename = "llama_model_7.bin";
    ifstream ifs(filename, ios::binary);
    if(ifs.fail()) {
        cout << "Could not load llama model.\n";
        return 1;
    }
    map<string, TensorFileInfo> tensorFileInfo = readTensorFileInfoTable(ifs);
    int numHeads, numKvHeads;
    float normEps;
    tie(numHeads, numKvHeads, normEps) = readParams(ifs);
    ifs.close();
    int maxSequenceLength = 1;
    int cacheSize = 500;
    LlamaModel model(tensorFileInfo, filename, numHeads, maxSequenceLength, cacheSize);
    list<int> tokenIdx{5000};
    model.evaluate(tokenIdx);
    return 0;
}