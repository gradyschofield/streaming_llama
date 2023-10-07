//
// Created by Grady Schofield on 10/7/23.
//

#ifndef STREAMING_LLAMA_LLAMAMODEL_H
#define STREAMING_LLAMA_LLAMAMODEL_H

#include<vector>

#include<Bf16.h>
#include<Common.h>
#include<NonTransformerWeights.h>
#include<TransformerBlock.h>

using namespace std;
using namespace Common;

class LLamaModelInterface {
public:
    virtual vector<float> evaluate(vector<int> const & tokens) = 0;
    virtual ~LLamaModelInterface() {}
};

template<typename T, Processor P>
class LlamaModel : public LLamaModelInterface {
    shared_ptr<NonTransformerWeights<T>> nonTransformerWeights;
    vector<shared_ptr<TransformerBlock<T, P>>> transformerBlocks;
    int tensorFile;
    shared_ptr<TransformerBlockScratch<T, P>> transformerBlockScratch;
    int numHeads;
    int numKvHeads;
    T normEps;
    FileStorageFormat fileStorageFormat;
    shared_ptr<Checker> checker;

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
        cout << "numHeads: " << numHeads << "\n";
        cout << "numKvHeads: " << numKvHeads << "\n";
        cout << "normEps: " << normEps << "\n";
    }

public:
    LlamaModel(string const & tensorFilename,
               int maxSequenceLength,
               int cacheSize,
               shared_ptr<Checker> checker = nullptr)
            : checker(checker)
    {
        ifstream ifs(tensorFilename, ios::binary);
        ifs.seekg(20);
        uint8_t storageType;
        ifs.read((char*)&storageType, 1);
        fileStorageFormat = intToFileStorageFormat(storageType);
        map<string, TensorFileInfo> tensorFileInfo = readTensorFileInfoTable(ifs);
        readParams(ifs);
        ifs.close();

        int layerCount = getLayerCount(tensorFileInfo);
        transformerBlockScratch = make_shared<TransformerBlockScratch<T, P>>(
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
        nonTransformerWeights = make_shared<NonTransformerWeights<T>>(tensorFileInfo, tensorFile, checker);
        for(int i = 0; i < layerCount; ++i) {
            transformerBlocks.push_back(make_shared<TransformerBlock<T, P>>(i,
                    tensorFileInfo,
                    tensorFile,
                    normEps,
                    nonTransformerWeights->getRopeFreqPtr(),
                    numHeads,
                    checker));
        }
    }

    ~LlamaModel() {
        close(tensorFile);
    }

    vector<float> evaluate(vector<int> const & tokens) override {
        int seqlen = tokens.size();
        Scratch<T> out = transformerBlockScratch->takeFreeIoPtr();
        nonTransformerWeights->getTokenEmbedding(tokens, out.getPtr());
        if(checker) {
            checker->submitResult(createDataAccessor(out.getPtr(),
                                                     {nonTransformerWeights->getTokenEmbeddings().getNumRows(),
                                                      seqlen},
                                                     out.getLeadingDimension()));
        }
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
        if(checker) {
            checker->finish();
        }
        T * outPtr = transformerBlockScratch->getOut().getPtr();
        for(int i = 0; i < nonTransformerWeights->getVocabularySize(); ++i) {
            ret[i] = outPtr[i];
        }
        return ret;
    }

};

template<Processor P>
shared_ptr<LLamaModelInterface> createLlamaModel(string filename,
                                                 int maxSequenceLength,
                                                 int cacheSize,
                                                 shared_ptr<Checker> checker = nullptr) {
    ifstream ifs(filename);
    ifs.seekg(20);
    uint8_t type;
    ifs.read((char*)&type, 1);
    ifs.close();
    FileStorageFormat fileStorageFormat = intToFileStorageFormat(type);
    switch(fileStorageFormat) {
        case Common::Bf16Aligned:
            return make_shared<LlamaModel<Bf16, P>>(filename, maxSequenceLength, cacheSize, checker);
        case Common::Fp32Aligned:
            return make_shared<LlamaModel<float, P>>(filename, maxSequenceLength, cacheSize, checker);
    }
    return nullptr; // unreachable
}

#endif //STREAMING_LLAMA_LLAMAMODEL_H
