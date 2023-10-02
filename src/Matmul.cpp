//
// Created by Grady Schofield on 10/1/23.
//

#include<cstdlib>
#include<list>
#include<unordered_map>

#include<Bf16.h>
#include<Common.h>
#include<Matmul.h>

using namespace std;
using namespace Common;

template<>
void multiplyMatrices<float>(const enum CBLAS_ORDER ORDER,
                             const enum CBLAS_TRANSPOSE TRANSA,
                             const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                             const int K, const float ALPHA, const float * A, const int LDA,
                             const float * B, const int LDB, const float BETA, float * C,
                             const int LDC) {
    cblas_sgemm(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
}

struct ScratchBuffer {
    int numRows;
    int numCols;
    float *buffer;
    uint64_t uses = 0;
};

namespace std {
    template<>
    struct hash<pair<int,int>> {
        size_t operator()(pair<int, int> const & t) const noexcept {
            return 31 * hash<int>{}(get<0>(t)) + hash<int>{}(get<1>(t));
        }
    };
}

static unordered_map<pair<int,int>, list<ScratchBuffer>> scratchBank;

static ScratchBuffer getScratch(int numRows, int numCols) {
    auto key = make_pair(numRows, numCols);
    if(!scratchBank[key].empty()) {
        ScratchBuffer ret = scratchBank.at(key).back();
        scratchBank[key].pop_back();
        ++ret.uses;
        return ret;
    } else {
        int lda = findAlignment(numRows, 64);
        void * p;
        posix_memalign(&p, 64, lda * numCols * sizeof(float));
        return ScratchBuffer{numRows, numCols, (float*)p};
    };
}

static void returnScratch(ScratchBuffer s) {
    scratchBank[make_pair(s.numRows, s.numCols)].push_back(s);
}

#ifdef __APPLE__
template<>
void multiplyMatrices<Bf16>(const enum CBLAS_ORDER ORDER,
                            const enum CBLAS_TRANSPOSE TRANSA,
                            const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                            const int K, const Bf16 ALPHA, const Bf16 * A, const int LDA,
                            const Bf16 * B, const int LDB, const Bf16 BETA, Bf16 * C,
                            const int LDC) {
    int aNumRows = TRANSA == CblasTrans ? K : M;
    int aNumCols = TRANSA == CblasTrans ? M : K;
    ScratchBuffer aScratch = getScratch(aNumRows, aNumCols);
    ScratchBuffer bScratch = getScratch(K, N);
    ScratchBuffer cScratch = getScratch(M, N);
    float * aBuffer = aScratch.buffer;
    float * bBuffer = bScratch.buffer;
    float * cBuffer = cScratch.buffer;
    for(int j = 0; j < aNumCols; ++j) {
        for(int i = 0; i < aNumRows; ++i) {
            aBuffer[i + aNumRows*j] = A[i + LDA*j].toFloat();
        }
    }
    for(int j = 0; j < N; ++j) {
        for (int i = 0; i < K; ++i) {
            bBuffer[i + K*j] = B[i + LDB*j].toFloat();
        }
    }
    cblas_sgemm(ORDER, TRANSA, TRANSB, M, N, K,
                ALPHA.toFloat(), aBuffer, aNumRows, bBuffer, K,
                BETA.toFloat(), cBuffer, M);
    for(int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            C[i + LDC*j] = cBuffer[i + M*j];
        }
    }
    returnScratch(aScratch);
    returnScratch(bScratch);
    returnScratch(cScratch);
}
#else
void multiplyMatrices<Bf16>(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                 const int K, const Bf16 ALPHA, const Bf16 * A, const int LDA,
                 const Bf16 * B, const int LDB, const Bf16 BETA, Bf16 * C,
                 const int LDC) {
    cblas_gemm_bf16_bf16_pack(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
}
#endif

