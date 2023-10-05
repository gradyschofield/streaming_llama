//
// Created by Grady Schofield on 9/22/23.
//

#include<Common.h>

#include<algorithm>
#include<set>
#include<sstream>

//ofstream logger("log");
using namespace std;

namespace Common {
    uint8_t fileStorageFormatToInt(FileStorageFormat outputFormat) {
        switch(outputFormat) {
            case Bf16Aligned:
                return 0;
            case Fp32Aligned:
                return 1;
            case Cuda:
                return 2;
        }
    }

    FileStorageFormat intToFileStorageFormat(uint8_t i) {
        switch(i) {
            case 0:
                return Bf16Aligned;
            case 1:
                return Fp32Aligned;
            case 2:
                return Cuda;
            default:
                throw 1;
        }
        return Bf16Aligned; // unreachable
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
}