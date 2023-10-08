//
// Created by Grady Schofield on 9/16/23.
//

#ifndef STREAMING_LLAMA_COMMON_H
#define STREAMING_LLAMA_COMMON_H

#include<cmath>
#include<cstdint>
#include<fstream>
#include<map>
#include<string>
#include<vector>

using namespace std;

//extern ofstream logger;

namespace Common {

    enum FileStorageFormat {
        Fp32Aligned,
        Bf16Aligned
    };

    enum Processor {
        Cpu,
        Gpu 
    };

    uint8_t fileStorageFormatToInt(FileStorageFormat fileStorageFormat);
    FileStorageFormat intToFileStorageFormat(uint8_t i);

    struct TensorFileInfo {
        int64_t offset;
        int numRows;
        int numColumns;
        int leadingDimension;
    };
    vector<pair<string, TensorFileInfo>> getTensorsForLayer(int layer, map<string, TensorFileInfo> const & tensorFileInfo);
    vector<pair<string, TensorFileInfo>> getNonTransformerBlockTensors(map<string, TensorFileInfo> const & tensorFileInfo);
    int getLayerCount(map<string, TensorFileInfo> const & tensorFileInfo);

    int findAlignment(int elements, int alignmentBytes);

    template<typename T>
    void layerNormalization(T * weights, T* src, int numRows, int leadingDimension, int seqlen, T normEps) {
        for(int j = 0; j < seqlen; ++j) {
            float accum = 0;
            T* ptr = &src[j * leadingDimension];
            for(int i = 0; i < numRows; ++i) {
                accum += (float)ptr[i] * (float)ptr[i];
            }
            float norm = 1.0 / sqrt(accum/numRows + normEps);
            for(int i = 0; i < numRows; ++i) {
                ptr[i] *= (float)weights[i] * norm;
            }
        }
    }

}


#endif //STREAMING_LLAMA_COMMON_H
