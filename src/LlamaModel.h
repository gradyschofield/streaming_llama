//
// Created by Grady Schofield on 10/7/23.
//

#ifndef STREAMING_LLAMA_LLAMAMODEL_H
#define STREAMING_LLAMA_LLAMAMODEL_H

#include<memory>
#include<vector>

#include<Bf16.h>
#include<Common.h>
#include<EvaluationTimings.h>
#include<NonTransformerWeights.h>
#include<TransformerBlock.h>

using namespace std;
using namespace Common;

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
    LlamaModelParams llamaModelParams;
    T normEps;
    FileStorageFormat fileStorageFormat;
    shared_ptr<Checker> checker;
    EvaluationTimings timings;


public:
    LlamaModel(string const & filename,
               int maxSequenceLength,
               int cacheSize,
               shared_ptr<Checker> checker = nullptr)
            : checker(checker)
    {
        cout << "Loading file " << filename << "\n";
        fileStorageFormat = readFileStorageFormat(filename);
        map<string, TensorFileInfo> tensorFileInfo = readTensorFileInfoTable(filename);
        llamaModelParams = readParams(filename);
        normEps = llamaModelParams.normEps;

        int layerCount = getLayerCount(tensorFileInfo);
        transformerBlockScratch = make_shared<TransformerBlockScratch<T>>(
                maxSequenceLength, cacheSize, llamaModelParams.numHeads,
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
        tensorFile = open(filename.c_str(), O_RDONLY);
        nonTransformerWeights = make_shared<NonTransformerWeights<T>>(tensorFileInfo, tensorFile, checker);
        for(int i = 0; i < layerCount; ++i) {
            transformerBlocks.push_back(make_shared<TransformerBlock<T>>(
                    i,
                    tensorFileInfo,
                    tensorFile,
                    normEps,
                    nonTransformerWeights->getRopeFreqPtr(),
                    llamaModelParams.numHeads,
                    checker));
        }
    }

    ~LlamaModel() {
        close(tensorFile);
    }

    vector<float> evaluate(vector<int> const & tokens) override {
        int seqlen = tokens.size();
        Scratch<T> out = transformerBlockScratch->takeFreeIoPtr();
        timings.start("Get token embeddings");
        nonTransformerWeights->getTokenEmbedding(tokens, out.getPtr());
        timings.finish("Get token embeddings");
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
                                             transformerBlockScratch,
                                             timings);
            transformerBlock->munmap();
        }
        vector<float> ret(nonTransformerWeights->getVocabularySize());
        Scratch<T> in = transformerBlockScratch->takeFreeIoPtr();
        timings.start("Compute output layer");
        memcpy(in.getPtr(),
               &out.getPtr()[out.getLeadingDimension() * (seqlen-1)],
               sizeof(T) * out.getLeadingDimension());
        nonTransformerWeights->applyOutputLayer(in,
                                                transformerBlockScratch->getOut(),
                                                1,
                                                normEps);
        timings.finish("Compute output layer");
        timings.print(cout);
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
            return make_shared<LlamaModel<Bf16>>(filename, maxSequenceLength, cacheSize, checker);
        case Common::Fp32Aligned:
            return make_shared<LlamaModel<float>>(filename, maxSequenceLength, cacheSize, checker);
    }
    return nullptr; // unreachable
}

#endif //STREAMING_LLAMA_LLAMAMODEL_H
