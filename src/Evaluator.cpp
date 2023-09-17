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

using namespace std;

class TokenEmbedding {
public:
    TokenEmbedding(map<string, TensorFileInfo> const & tensorFileInfo){
    }
};

class Weights {
    void * ptr;
    int numRows;
    int numColumns;
public:
    void * getPtr() const {
        return ptr;
    }

    int getNumRow() const {
        return numRows;
    }

    int getNumColumns() const {
        return numColumns;
    }
};

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

public:
    TransformerBlock(int layer, map<string, TensorFileInfo> const & tensorFileInfo){
    }

    void evaluate(void * p, int seqlen) {
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
    }
};

class OutputBlock {
public:
    OutputBlock(map<string, TensorFileInfo> const & tensorFileInfo){
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

class LlamaModel {
    shared_ptr<TokenEmbedding> tokenEmbedding;
    vector<shared_ptr<TransformerBlock>> transformerBlocks;
    shared_ptr<OutputBlock> outputBlock;
public:
    LlamaModel(map<string, TensorFileInfo> const & tensorFileInfo){
        tokenEmbedding = make_shared<TokenEmbedding>(tensorFileInfo);
        int layerCount = getLayerCount(tensorFileInfo);
        for(int i = 0; i < layerCount; ++i) {
            transformerBlocks.push_back(make_shared<TransformerBlock>(i, tensorFileInfo));
        }
        outputBlock = make_shared<OutputBlock>(tensorFileInfo);
    }
    vector<float> evaluate(list<int> const & tokens) {
        vector<float> t;
        return t;
    }
};


map<string, TensorFileInfo> readTensorFileInfoTable(ifstream & ifs) {
    uint64_t tensorOffsetTablePos = 0;
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

int main(int argc, char ** argv) {
    ifstream ifs("llama_model_7.bin", ios::binary);
    if(ifs.fail()) {
        cout << "Could not load llama model.\n";
        return 1;
    }
    map<string, TensorFileInfo> tensorFileInfo = readTensorFileInfoTable(ifs);
    auto tfi = getTensorsForLayer(0, tensorFileInfo);
    for(auto & p : tfi) {
        cout << p.first << " " << p.second.offset << "\n";
    }
    LlamaModel model(tensorFileInfo);
    return 0;
}