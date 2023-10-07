#include<unistd.h>

#include<fstream>
#include<iostream>
#include<iomanip>
#include<list>
#include<map>
#include<set>
#include<sstream>
#include<string>
#include<unordered_map>
#include<vector>

#include<Common.h>
#include<Exception.h>

using namespace std;
using namespace Common;

// TODO: move map from output format to int to common.h/cpp, read otuput format in Evaluator and deal with it properl

map<string, int64_t> readTensorOffsetTable(ifstream & ifs) {
    int64_t tensorOffsetTablePos;
    ifs.seekg(0);
    ifs.read((char*)&tensorOffsetTablePos, 8);
    ifs.seekg(tensorOffsetTablePos);
    int numTensors;
    ifs.read((char*)&numTensors, 4);
    map<string, int64_t> ret;
    for(int i = 0; i < numTensors; ++i) {
        int strLen;
        ifs.read((char*)&strLen, 4);
        vector<char> buffer(strLen);
        ifs.read(buffer.data(), strLen);
        int64_t offset;
        ifs.read((char*)&offset, 8);
        ret.emplace(string(buffer.data(), buffer.size()), offset);
    }
    return ret;
}

enum ParallelizationLayout {
    SplitColumns,
    SplitRows,
    Duplicated
};

unordered_map<string, ParallelizationLayout> TENSOR_LAYOUT {
        {"attention.wq.weight", SplitColumns},
        {"attention.wk.weight", SplitColumns},
        {"attention.wv.weight", SplitColumns},
        {"attention.wo.weight", SplitRows},
        {"tok_embeddings.weight", SplitRows},
        {"feed_forward.w1.weight", SplitColumns},
        {"feed_forward.w2.weight", SplitRows},
        {"feed_forward.w3.weight", SplitColumns},
        {"output.weight", SplitColumns},
};

struct Fp32 {
    union {
        float x;
        uint32_t i;
    };

    Fp32 & operator=(uint16_t t) {
        this->i = t;
        this->i <<= 16;
        return *this;
    }

    Fp32 & operator=(float t) {
        this->x = t;
        return *this;
    }

    ostream & printBinary(ostream & os) {
        for(int i = 31; i >= 0; --i) {
            os << (this->i & (1<<i) ? 1 : 0);
        }
        return os;
    }
};

template<typename T>
pair<string, TensorFileInfo> consolidateTensorFragments(map<string, int64_t> const & fragments,
                                                        ParallelizationLayout parallelizationLayout,
                                                        FileStorageFormat fileStorageFormat,
                                                        ifstream & ifs, ofstream & ofs) {
    int64_t position = ofs.tellp();
    int numRows = 0;
    int numColumns = 0;
    int largestSplitDim = 0;
    //Inspect all the tensor dimensions to determine the consolidated numRows/numColumns
    for(auto & p : fragments) {
        ifs.seekg(p.second);
        int rows, cols;
        ifs.read((char*) &rows, 4);
        ifs.read((char*) &cols, 4);
        if(parallelizationLayout == SplitColumns) {
            numRows += rows;
            numColumns = cols;
            largestSplitDim = max(largestSplitDim, rows);
        } else {
            numRows = rows;
            numColumns += cols;
            largestSplitDim = max(largestSplitDim, cols);
        }
    }
    int leadingDim = 0;
    if (fileStorageFormat == Fp32Aligned || fileStorageFormat == Bf16Aligned) {
        leadingDim = findAlignment(numRows, 64);
    } else {
        cout << "Need to handle this fileStorageFormat in consolidateTensorFragments\n";
        exit(1);
    }
    int largestSize = 0;
    if (parallelizationLayout == SplitColumns) {
        largestSize = numColumns * largestSplitDim;
    } else {
        largestSize = numRows * largestSplitDim;
    }
    vector<uint16_t> tmp(largestSize);
    vector<T> storage(leadingDim * numColumns);
    int offset = 0;
    for(auto & p : fragments) {
        ifs.seekg(p.second);
        int rows, cols;
        ifs.read((char *) &rows, 4);
        ifs.read((char *) &cols, 4);
        ifs.read((char *) tmp.data(), rows * cols * 2);
        if (parallelizationLayout == SplitColumns) {
            for(int j = 0; j < cols; ++j) {
                for(int i = 0; i < rows; ++i) {
                    storage[offset + i + j*leadingDim] = tmp[i + j*rows];
                }
            }
            offset += rows;
        } else {
            for(int j = 0; j < cols; ++j) {
                for(int i = 0; i < rows; ++i) {
                    storage[i + (j+offset)*leadingDim] = tmp[i + j*rows];
                }
            }
            offset += cols;
        }
    }
    string tensorName = begin(fragments)->first.substr(3);
    if (tensorName == "tok_embeddings.weight") {
        //transpose the token embedding matrix
        int newNumRows = numColumns;
        int newNumColumns = numRows;
        int newLeadingDim = 0;
        if (fileStorageFormat == Fp32Aligned || fileStorageFormat == Bf16Aligned) {
            newLeadingDim = findAlignment(newNumRows, 64);
        } else {
            cout << "Need to handle this fileStorageFormat in consolidateTensorFragments\n";
            exit(1);
        }
        vector<T> newStorage(newLeadingDim * newNumColumns);
        for(int i = 0; i < leadingDim; ++i) {
            int newJ = i;
            for(int j = 0; j < numColumns; ++j) {
                int newI = j;
                newStorage[newI + newJ*newLeadingDim] = storage[i + j*leadingDim];
            }
        }
        numColumns = newNumColumns;
        numRows = newNumRows;
        leadingDim = newLeadingDim;
        swap(storage, newStorage);
    }
    ofs.write((char*)storage.data(), numColumns * leadingDim * sizeof(T));
    cout << "Wrote consolidated tensor " << tensorName << " rows: " << numRows <<
        " cols: " << numColumns << " ld: " << leadingDim << "\n";
    return make_pair(tensorName, TensorFileInfo{position, numRows, numColumns, leadingDim});
}

template<typename T>
pair<string, TensorFileInfo> writeNonParallelizedTensor(string const & tensorName,
                                                        int64_t tensorOffset,
                                                        FileStorageFormat fileStorageFormat,
                                                        ifstream & ifs,
                                                        ofstream & ofs) {
    int64_t position = ofs.tellp();
    int rows, cols;
    ifs.seekg(tensorOffset);
    ifs.read((char *) &rows, 4);
    ifs.read((char *) &cols, 4);
    vector<uint16_t> tmp(rows*cols);
    ifs.read((char *) tmp.data(), rows * cols * 2);
    int leadingDim = 0;
    if (fileStorageFormat == Fp32Aligned || fileStorageFormat == Bf16Aligned) {
        leadingDim = findAlignment(rows, 64);
    } else {
        cout << "Need to handle this fileStorageFormat in consolidateTensorFragments\n";
        exit(1);
    }
    vector<T> storage(leadingDim*cols);
    for(int j = 0; j < cols; ++j) {
        for(int i = 0; i < rows; ++i) {
            storage[i + j*leadingDim] = tmp[i + j*rows];
        }
    }
    ofs.write((char*)storage.data(), storage.size()*sizeof(T));
    string finalTensorName = tensorName.substr(3);
    cout << "Wrote duplicated tensor " << finalTensorName << " rows: " <<
        rows << " cols: " << cols << " ld: " << rows << "\n";
    return make_pair(finalTensorName,
                     TensorFileInfo{position, rows, cols, rows});
}

int countParallelFragments(map<string, int64_t> const & offsetTable) {
    set<string> prefixes;
    for(auto & p : offsetTable) {
        prefixes.insert(p.first.substr(0, 2));
    }
    return prefixes.size();
}

int countLayers(map<string, int64_t> const & offsetTable) {
    set<string> layers;
    for(auto & p : offsetTable) {
        string t = p.first.substr(3);
        if(t.starts_with("layers.")) {
            string t2 = t.substr(7);
            int idx = t2.find('.');
            layers.insert(t2.substr(0, idx));
        }
    }
    return layers.size();
}

list<string> getTensorNamesInOrder(map<string, int64_t> const & offsetTable) {
    int numParallelFragments = countParallelFragments(offsetTable);
    int numLayers = countLayers(offsetTable);
    list<string> tensorNames;
    for(int fragment = 0; fragment < numParallelFragments; ++fragment) {
        auto printStr = [](int fragment, string tail) {
            stringstream sstr;
            sstr << setw(2) << setfill('0') << fragment << "." << tail;
            return sstr.str();
        };
        tensorNames.push_back(printStr(fragment, "tok_embeddings.weight"));
        tensorNames.push_back(printStr(fragment, "rope.freqs"));
        tensorNames.push_back(printStr(fragment, "norm.weight"));
        tensorNames.push_back(printStr(fragment, "output.weight"));
        for(int layer = 0; layer < numLayers; ++layer) {
            auto printStr2 = [](int fragment, int layer, string tail) {
                stringstream sstr;
                sstr << setw(2) << setfill('0') << fragment << ".layers." << layer << "." << tail;
                return sstr.str();
            };
            tensorNames.push_back(printStr2(fragment, layer, "attention_norm.weight"));
            tensorNames.push_back(printStr2(fragment, layer, "attention.wq.weight"));
            tensorNames.push_back(printStr2(fragment, layer, "attention.wk.weight"));
            tensorNames.push_back(printStr2(fragment, layer, "attention.wv.weight"));
            tensorNames.push_back(printStr2(fragment, layer, "attention.wo.weight"));
            tensorNames.push_back(printStr2(fragment, layer, "ffn_norm.weight"));
            tensorNames.push_back(printStr2(fragment, layer, "feed_forward.w1.weight"));
            tensorNames.push_back(printStr2(fragment, layer, "feed_forward.w2.weight"));
            tensorNames.push_back(printStr2(fragment, layer, "feed_forward.w3.weight"));
        }
    }
    return tensorNames;
}

bool shouldPrependWithPad(string const & tensorName) {
    return tensorName.ends_with("tok_embeddings.weight") || tensorName.ends_with("attention_norm.weight");
}

void writePad(ofstream & ofs) {
    int64_t cur = ofs.tellp();
    int64_t pageSize = getpagesize();
    int64_t excess = cur % pageSize;
    if(excess) {
        int64_t padSize = pageSize - excess;
        vector<int8_t> dummy(padSize);
        ofs.write((char *) dummy.data(), padSize);
    }
}

void transferParamsToOutFile(ifstream & ifs, ofstream & ofs) {
    ifs.seekg(8);
    int n_heads = 0;
    int n_kv_heads = 0;
    float norm_eps = 0;
    ifs.read((char*) &n_heads, 4);
    ifs.read((char*) &n_kv_heads, 4);
    ifs.read((char*) &norm_eps, 4);
    ofs.write((char*) &n_heads, 4);
    ofs.write((char*) &n_kv_heads, 4);
    ofs.write((char*) &norm_eps, 4);
}

void writeOutputFormat(ofstream & ofs, FileStorageFormat fileStorageFormat) {
    uint8_t dummy = fileStorageFormatToInt(fileStorageFormat);
    ofs.write((char*)&dummy, 1);
}

FileStorageFormat stringToFileStorageFormat(string s) {
    if(s == "bf16") {
        return Bf16Aligned;
    } else if(s == "fp32") {
        return Fp32Aligned;
    } else {
        stringstream sstr;
        sstr << "The file storage string, " << s << ", was neither bf16 nor fp32.  Do you need to add code to handle this format?";
        throw Exception(sstr.str());
    }
}

int main(int argc, char ** argv) {
    ifstream ifs(argv[2]);
    ofstream ofs("llama_model.bin");
    int64_t dummy = 0;
    FileStorageFormat fileStorageFormat = stringToFileStorageFormat(argv[1]);
    ofs.write((char *) &dummy, 8);
    map<string, int64_t> offsetTable = readTensorOffsetTable(ifs);
    transferParamsToOutFile(ifs, ofs);
    writeOutputFormat(ofs, fileStorageFormat);
    list<pair<string, TensorFileInfo>> tensorInfoTable;
    list<string> tensorNames = getTensorNamesInOrder(offsetTable);
    for(string const & tensorName : tensorNames) {
        int64_t tensorOffset = offsetTable.at(tensorName);
        bool isParallel = false;
        ParallelizationLayout parallelizationLayout = Duplicated;
        for(auto & p2: TENSOR_LAYOUT) {
            if(tensorName.ends_with(p2.first)) {
                isParallel = true;
                parallelizationLayout = p2.second;
                break;
            }
        }
        if(tensorName.starts_with("00")) {
            if(shouldPrependWithPad(tensorName)) {
                writePad(ofs);
            }
            if (!isParallel) {
                //Should be able to write this data and forget about parallel copies.  But do confirm that the parallel
                //copies are identical or close.
                pair<string, TensorFileInfo> offsetInfo;
                if(fileStorageFormat == Fp32Aligned) {
                    offsetInfo = writeNonParallelizedTensor<Fp32>(tensorName, tensorOffset,
                                                                  fileStorageFormat, ifs, ofs);
                } else {
                    offsetInfo = writeNonParallelizedTensor<uint16_t>(tensorName, tensorOffset,
                                                                  fileStorageFormat, ifs, ofs);
                }
                tensorInfoTable.push_back(std::move(offsetInfo));
            } else {
                string tensorNameNoPrefix = tensorName.substr(3);
                //collect the names of all fragments of the tensor
                map<string, int64_t> fragments;
                fragments.emplace(tensorName, tensorOffset);
                for (auto &p2: offsetTable) {
                    if (p2.first.ends_with(tensorNameNoPrefix)) {
                        fragments.emplace(p2);
                    }
                }
                pair<string, TensorFileInfo> offsetInfo;
                if(fileStorageFormat == Fp32Aligned) {
                    offsetInfo = consolidateTensorFragments<Fp32>(fragments, parallelizationLayout,
                                                                  fileStorageFormat, ifs, ofs);
                } else {
                    offsetInfo = consolidateTensorFragments<uint16_t>(fragments, parallelizationLayout,
                                                                  fileStorageFormat, ifs, ofs);
                }
                tensorInfoTable.push_back(std::move(offsetInfo));
            }
        }
    }
    int64_t tensorInfoTablePosition = ofs.tellp();
    int numTensors = tensorInfoTable.size();
    ofs.write((char*) &numTensors, 4);
    for(auto & p : tensorInfoTable) {
        int nameLen = p.first.size();
        ofs.write((char*) &nameLen, 4);
        ofs.write((char*) p.first.data(), nameLen);
        ofs.write((char*) &p.second.offset, 8);
        ofs.write((char*) &p.second.numRows, 4);
        ofs.write((char*) &p.second.numColumns, 4);
        ofs.write((char*) &p.second.leadingDimension, 4);
    }
    ofs.seekp(0);
    ofs.write((char*) &tensorInfoTablePosition, 8);
    ofs.close();
    return 0;
}
