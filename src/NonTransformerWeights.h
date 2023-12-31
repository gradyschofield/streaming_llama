//
// Created by Grady Schofield on 10/7/23.
//

#ifndef STREAMING_LLAMA_NONTRANSFORMERWEIGHTS_H
#define STREAMING_LLAMA_NONTRANSFORMERWEIGHTS_H

#include<unistd.h>
#include<sys/mman.h>
#include<fcntl.h>

#include<memory>

#include<Checker.h>
#include<LayerNormalization.h>
#include<Matmul.h>
#include<Scratch.h>
#include<Weights.h>

using namespace std;

template<typename T>
class NonTransformerWeights {
    unique_ptr<Weights<T>> tokenEmbeddings;
    unique_ptr<Weights<T>> ropeFreqs;
    unique_ptr<Weights<T>> outputNormalizers;
    unique_ptr<Weights<T>> outputWeights;
    off_t mapOffset;
    size_t mapLength;
    void * mapAddress = nullptr;
    shared_ptr<Checker> checker;

public:
    NonTransformerWeights(
            map<string, TensorFileInfo> const & tensorFileInfo,
            int tensorFile,
            shared_ptr<Checker> checker = nullptr)
        : checker(checker)
    {
        vector<pair<string, TensorFileInfo>> tensorInfos = getNonTransformerBlockTensors(tensorFileInfo);
        mapOffset = tensorInfos.front().second.offset;
        TensorFileInfo const & tfi = tensorInfos.back().second;
        mapLength = tfi.offset + tfi.leadingDimension * tfi.numColumns * sizeof(T) - mapOffset;
        mapAddress = mmap(nullptr, mapLength, PROT_READ, MAP_PRIVATE, tensorFile, mapOffset);
        if(mapAddress == MAP_FAILED) {
            stringstream sstr;
            sstr << "mmap failed for offset " << mapOffset << " and length " << mapLength;
            throw Exception(sstr.str());
        }
        map<string, TensorFileInfo> tensorInfoMap;
        for(auto & p : tensorInfos) {
            string const & tensorName = p.first;
            tensorInfoMap.emplace(tensorName, p.second);
        }
        tokenEmbeddings = make_unique<Weights<T>>(mapAddress, mapOffset, tensorInfoMap.at("tok_embeddings.weight"));
        ropeFreqs = make_unique<Weights<T>>(mapAddress, mapOffset, tensorInfoMap.at("rope.freqs"));
        outputNormalizers = make_unique<Weights<T>>(mapAddress, mapOffset, tensorInfoMap.at("norm.weight"));
        outputWeights = make_unique<Weights<T>>(mapAddress, mapOffset, tensorInfoMap.at("output.weight"));
        munmap(mapAddress, mapLength);
        mapAddress = nullptr;
    }

    ~NonTransformerWeights() {
    }

    T * getRopeFreqPtr() {
        return ropeFreqs->getPtr();
    }

    int getVocabularySize() const {
        return tokenEmbeddings->getNumColumns();
    }

    void getTokenEmbedding(vector<int> const & tokens, T * out) {
        T const * ptr = tokenEmbeddings->getPtr();
        int i = 0;
        for (int tok : tokens) {
            memcpy(&out[i* tokenEmbeddings->getLeadingDimension()],
                   &ptr[tok * tokenEmbeddings->getLeadingDimension()],
                   tokenEmbeddings->getNumRows() * sizeof(T));
            ++i;
        }
    }

    Weights<T> const * getTokenEmbeddings() {
        return tokenEmbeddings.get();
    }

    void applyOutputLayer(Scratch<T> * in, Scratch<T> * out, int seqlen, float normEps) {
        LayerNormalization<T>::exec(outputNormalizers->getPtr(),
                                       in->getPtr(),
                                       outputWeights->getNumColumns(),
                                       in->getLeadingDimension(),
                                       seqlen,
                                       normEps);
        if (checker) {
            checker->submitResult(createDataAccessor(in->getPtr(),
                                                     {outputWeights->getNumColumns(),
                                                      seqlen},
                                                     in->getLeadingDimension()));
        }

        multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                               outputWeights->getNumRows(), seqlen, outputWeights->getNumColumns(),
                               1.0,
                               outputWeights->getMetalBuffer(),
                               outputWeights->getLeadingDimension(),
                               in->getMetalBuffer(),
                               in->getLeadingDimension(),
                               0.0,
                               out->getMetalBuffer(),
                               out->getLeadingDimension());
        Metal::waitUntilCompleted(0, reclaimMatvecBuffers);
        if(checker) {
            checker->submitResult(createDataAccessor(out->getPtr(),
                                                     {outputWeights->getNumRows(),
                                                      seqlen},
                                                     out->getLeadingDimension()));
        }
    }
};

#endif //STREAMING_LLAMA_NONTRANSFORMERWEIGHTS_H
