//
// Created by Grady Schofield on 9/22/23.
//

#include<algorithm>
#include<set>
#include<sstream>

#include<Common.h>
#include<Exception.h>

//ofstream logger("log");
using namespace std;

namespace Common {
    uint8_t fileStorageFormatToInt(FileStorageFormat outputFormat) {
        switch(outputFormat) {
            case Bf16Aligned:
                return 0;
            case Fp32Aligned:
                return 1;
            case Bf16Unaligned:
                return 2;
        }
        return 255; // unreachable
    }

    FileStorageFormat intToFileStorageFormat(uint8_t i) {
        switch(i) {
            case 0:
                return Bf16Aligned;
            case 1:
                return Fp32Aligned;
            case 2:
                return Bf16Unaligned;
            default:
                stringstream sstr;
                sstr << "The file format integer, " << (int)i << ", was neither 0 or 1.  Do you need to add code for this format?";
                throw Exception(sstr.str());
        }
        return Bf16Aligned; // unreachable
    }

    vector<pair<string, TensorFileInfo>> getTensorsForLayer(int layer, map<string, TensorFileInfo> const & tensorFileInfo) {
        stringstream sstr;
        sstr << "layers." << layer << ".";
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

    int getLayerCount(map<string, TensorFileInfo> const & tensorFileInfo) {
        set<string> layers;
        for(auto & p : tensorFileInfo) {
            if(p.first.starts_with("layers.")) {
                layers.insert(p.first.substr(7, p.first.find('.', 7)-7));
            }
        }
        return layers.size();
    }

    int findAlignment(int elements, int alignmentBytes) {
        int bytesPastAlignment = (elements * 4) % alignmentBytes;
        if (bytesPastAlignment == 0) return elements;
        else return (1 + ((elements * 4) / alignmentBytes)) * alignmentBytes / 4;
    }

    map<string, TensorFileInfo> readTensorFileInfoTable(string filename) {
        ifstream ifs(filename, ios::binary);
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

    LlamaModelParams readParams(string filename) {
        ifstream ifs(filename, ios::binary);
        LlamaModelParams llamaModelParams;
        ifs.seekg(8);
        ifs.read((char*) &llamaModelParams.numHeads, 4);
        ifs.read((char*) &llamaModelParams.numKvHeads, 4);
        ifs.read((char*) &llamaModelParams.normEps, 4);
        return llamaModelParams;
    }

    FileStorageFormat readFileStorageFormat(string filename) {
        ifstream ifs(filename, ios::binary);
        ifs.seekg(20);
        uint8_t storageType;
        ifs.read((char*)&storageType, 1);
        return intToFileStorageFormat(storageType);
    }
}
