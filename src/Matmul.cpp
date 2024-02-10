//
// Created by Grady Schofield on 10/1/23.
//

#include<atomic>
#include<barrier>
#include<condition_variable>
#include<cstdlib>
#include<functional>
#include<iostream>
#include<list>
#include<mutex>
#include<set>
#include<thread>
#include<unordered_map>

#include<Bf16.h>
#include<Common.h>
#include<Matmul.h>
#include<MetalHelpers.h>


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
    int leadingDimension;
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

static thread_local unordered_map<pair<int,int>, list<ScratchBuffer>> scratchBank;

static ScratchBuffer getScratch(int numRows, int numCols) {
    auto key = make_pair(numRows, numCols);
    if(!scratchBank[key].empty()) {
        ScratchBuffer ret = scratchBank.at(key).back();
        scratchBank[key].pop_back();
        ++ret.uses;
        return ret;
    } else {
        int leadingDimension = findAlignment(numRows, 64);
        void * p;
        posix_memalign(&p, 64, leadingDimension * numCols * sizeof(float));
        return ScratchBuffer{numRows, numCols, leadingDimension, (float*)p};
    };
}

static void returnScratch(ScratchBuffer s) {
    scratchBank[make_pair(s.numRows, s.numCols)].push_back(s);
}

struct WorkParams {
    int M;
    int N;
    int K;
    CBLAS_TRANSPOSE TRANSA;
    Bf16 const * A;
    int LDA;
    Bf16 const * B;
    int startCol;
    int endCol;
    bool finished = true;
};

struct Matvec {
    vector<thread> threads;
    vector<float*> partialOut;
    vector<atomic<int>> ci;
    atomic<int> barrierCompletionIndicator;
    atomic<int> barrierParamsWritten;
    atomic<Bf16*> staticC;
    atomic<int> staticM;
    atomic<int> numThreads;
    barrier<function<void()>> bar;
    vector<WorkParams> params;
    vector<mutex> workerWaitConditionVarMutex;
    mutex mainWaitConditionVarMutex;
    condition_variable mainWaitConditionVar;
    vector<condition_variable> workerWaitConditionVar;
    mutex coutLock;

public:
    void accumulate() {
        Bf16 * C = staticC.load(memory_order_acquire);
        int M = staticM.load(memory_order_acquire);
        for (int i = 0; i < numThreads; ++i) {
            atomic_load_explicit(&ci[i], memory_order_relaxed);
        }
        atomic_thread_fence(memory_order_acquire);
        float * tmp = partialOut[0];
        for(int j = 1; j < numThreads; ++j) {
            float * out = partialOut[j];
            for (int i = 0; i < M; ++i) {
                tmp[i] += out[i];
            }
        }
        for (int i = 0; i < M; ++i) {
            C[i] = tmp[i];
        }
        atomic_thread_fence(memory_order_release);
        atomic_store_explicit(&barrierCompletionIndicator, 1, memory_order_relaxed);
        {
            lock_guard<mutex> lk(mainWaitConditionVarMutex);
            for (WorkParams & wp : params) wp.finished = true;
        }
        //signal main thread for completion
        mainWaitConditionVar.notify_all();
    }

    Matvec(int numThreads)
            : partialOut(numThreads),
              ci(numThreads),
              numThreads(numThreads),
              bar(numThreads, bind(&Matvec::accumulate, this)),
              params(numThreads),
              workerWaitConditionVarMutex(numThreads),
              workerWaitConditionVar(numThreads)
    {
        auto worker = [this](int threadIdx) {
            while (true) {
                /*
                    It's possible the barrier accumulator finished and the main thread notified for
                    another matmul before the worker got to this point.  So test to see if params is nonempty.
                 */
                unique_lock<mutex> lk(workerWaitConditionVarMutex[threadIdx]);
                if (params[threadIdx].finished) {
                    // wait for condition
                    workerWaitConditionVar[threadIdx].wait(lk);
                    //hande spurious wake up
                }
                /*
                atomic_load_explicit(&ci[threadIdx], memory_order_relaxed);
                atomic_thread_fence(memory_order_acquire);
                 */
                WorkParams wp = params[threadIdx];
                int M = wp.M;
                int N = wp.N;
                int K = wp.K;
                CBLAS_TRANSPOSE TRANSA = wp.TRANSA;
                Bf16 const *A = wp.A;
                int LDA = wp.LDA;
                Bf16 const *B = wp.B;
                int startCol = wp.startCol;
                int endCol = wp.endCol;
                ScratchBuffer cScratch = getScratch(M, N);
                float *cBuffer = cScratch.buffer;
                partialOut[threadIdx] = cBuffer;
                for (int i = 0; i < M; ++i) {
                    cBuffer[i] = 0;
                }
                if (TRANSA == CblasTrans) {
                    for (int i = 0; i < M; ++i) {
                        for (int j = startCol; j < endCol; ++j) {
                            cBuffer[i] += A[j + LDA * i].toFloat() * B[j].toFloat();
                        }
                    }
                } else {
                    for (int j = startCol; j < endCol; ++j) {
                        float b = B[j].toFloat();
                        for (int i = 0; i < M; ++i) {
                            cBuffer[i] += A[i + LDA * j].toFloat() * b;
                        }
                    }
                }
                atomic_thread_fence(memory_order_release);
                atomic_store_explicit(&ci[threadIdx], 1, memory_order_relaxed);
                bar.arrive_and_wait();
                returnScratch(cScratch);
            }
        };
        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(worker, i);
        }
    }

    void run(int M, int N, int K, CBLAS_TRANSPOSE TRANSA, Bf16 const * A, int LDA, Bf16 const * B, Bf16 * C) {
        atomic_store_explicit(&staticC, C, memory_order_release);
        atomic_store_explicit(&staticM, M, memory_order_release);

        {
            int startCol = 0;
            for (int i = 0; i < numThreads; ++i) {
                {
                    lock_guard lg(workerWaitConditionVarMutex[i]);
                    int cols = (K / numThreads) + (i < K % numThreads ? 1 : 0);
                    int endCol = startCol + cols;
                    params[i] = WorkParams{M, N, K, TRANSA, A, LDA, B, startCol, endCol, false};
                    startCol = endCol;
                }
                workerWaitConditionVar[i].notify_all();
            }
            /*
            atomic_thread_fence(memory_order_release);
            for (int i = 0; i < numThreads; ++i) {
                atomic_store_explicit(&ci[i], 1, memory_order_relaxed);
            }
             */
        }
        //signal all threads
        //wait for all threads
        unique_lock<mutex> lk(mainWaitConditionVarMutex);
        if (!params[0].finished) {
            mainWaitConditionVar.wait(lk);
        }
        atomic_load_explicit(&barrierCompletionIndicator, memory_order_relaxed);
        atomic_thread_fence(memory_order_acquire);
    }
};

set<tuple<int,int>> leftInput;
set<tuple<int,int>> rightInput;
set<tuple<int,int>> output;
long numCalls = 0;

template<>
void multiplyMatrices<Bf16>(const enum CBLAS_ORDER ORDER,
                            const enum CBLAS_TRANSPOSE TRANSA,
                            const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                            const int K, const Bf16 ALPHA, const Bf16 * A, const int LDA,
                            const Bf16 * B, const int LDB, const Bf16 BETA, Bf16 * C,
                            const int LDC) {
    int aNumRows = TRANSA == CblasTrans ? K : M;
    int aNumCols = TRANSA == CblasTrans ? M : K;
    if (N > 1 || TRANSA == CblasTrans || LDA != M || LDB != K || LDC != M) {
        ScratchBuffer aScratch = getScratch(aNumRows, aNumCols);
        ScratchBuffer bScratch = getScratch(K, N);
        ScratchBuffer cScratch = getScratch(M, N);
        float *aBuffer = aScratch.buffer;
        float *bBuffer = bScratch.buffer;
        float *cBuffer = cScratch.buffer;
        for (int j = 0; j < aNumCols; ++j) {
            for (int i = 0; i < aNumRows; ++i) {
                aBuffer[i + aScratch.leadingDimension * j] = A[i + LDA * j].toFloat();
            }
        }
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < K; ++i) {
                bBuffer[i + bScratch.leadingDimension * j] = B[i + LDB * j].toFloat();
            }
        }
        /*
        leftInput.insert(make_tuple(aNumRows, aNumCols));
        rightInput.insert(make_tuple(K, N));
        output.insert(make_tuple(M, N));
         */
        cblas_sgemm(ORDER, TRANSA, TRANSB, M, N, K,
                    ALPHA.toFloat(), aBuffer, aScratch.leadingDimension,
                    bBuffer, bScratch.leadingDimension,
                    BETA.toFloat(), cBuffer, cScratch.leadingDimension);
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                C[i + LDC * j] = cBuffer[i + cScratch.leadingDimension * j];
            }
        }
        returnScratch(aScratch);
        returnScratch(bScratch);
        returnScratch(cScratch);
    } else {
        static Matvec * matvec = nullptr;
        int numThreads = 8;
        if (!matvec) {
            matvec = new Matvec(numThreads);
        }
        matvec->run(M, N, K, TRANSA, A, LDA, B, C);
    }
}

static multimap<pair<int, int>, MTL::Buffer * > freeMatvecTemporaryBuffers;
static multimap<pair<int, int>, MTL::Buffer * > inUseMatvecTemporaryBuffers;
static Metal::Function * matvecFunc = nullptr;
static Metal::Function * reducerFunc = nullptr;

static void initMatvec() {
    static string matvecSrc = R"(
            #include <metal_stdlib>
            using namespace metal;

            kernel void singleColumnMatvec(device const bfloat4 * mat [[buffer(0)]],
                                        device const bfloat * vec [[buffer(1)]],
                                        device float4 * result [[buffer(2)]],
                                        constant int64_t & numRows [[buffer(3)]],
                                        constant int64_t & numCols [[buffer(4)]],
                                        uint2 threadPos [[thread_position_in_threadgroup]],
                                        uint2 threadDim [[threads_per_threadgroup]],
                                        uint2 groupPos [[threadgroup_position_in_grid]],
                                        uint2 groupDim [[threadgroups_per_grid]] ) {
                int64_t numRows_4 = numRows >> 2;
                int64_t threadRowOffset = threadPos.x + groupPos.x * threadDim.x;
                float4 t = 0;
                for (uint j = 0; j < numCols; j+=groupDim.y) {
                    t += static_cast<float4>(mat[threadRowOffset + (j+groupPos.y) * numRows_4] * vec[j+groupPos.y]);
                }
                result[threadRowOffset + groupPos.y * numRows_4] = t;
            }
        )";

    static string reducerSrc = R"(
            #include <metal_stdlib>
            using namespace metal;

            kernel void singleColumnMatvecReducer(
                        device float4 * input [[buffer(0)]],
                        device bfloat4 * result [[buffer(1)]],
                        constant int64_t & numRows [[buffer(2)]],
                        constant int64_t & numTemporaries [[buffer(3)]],
                        constant int64_t & outputOffsetElements [[buffer(4)]],
                        uint2 threadPos [[thread_position_in_threadgroup]],
                        uint2 threadDim [[threads_per_threadgroup]],
                        uint2 groupPos [[threadgroup_position_in_grid]],
                        uint2 groupDim [[threadgroups_per_grid]] ) {
                int64_t numRows4 = numRows >> 2;
                int64_t threadRowOffset = threadPos.x + groupPos.x * threadDim.x;
                float4 t = 0;
                for (uint j = 0; j < numTemporaries; ++j) {
                    t += input[threadRowOffset + j * numRows4];
                }
                result[(outputOffsetElements >> 2) + threadRowOffset] = static_cast<bfloat4>(t);
            }
        )";

    matvecFunc = Metal::getFunction(matvecSrc, "singleColumnMatvec");
    reducerFunc = Metal::getFunction(reducerSrc, "singleColumnMatvecReducer");
}

void matvec(MTL::Buffer * mat,
            MTL::Buffer * in,
            MTL::Buffer * out,
            long numRows,
            long numCols,
            long outputOffsetElements) {

    if (!matvecFunc) {
        initMatvec();
    }

    long stripLength = 32;
    long stripDimGroups = 8;
    long stripsPerGroup = 1;

    pair<int, int> tmpKey = make_pair(numRows, stripDimGroups);
    auto iter = freeMatvecTemporaryBuffers.find(tmpKey);
    MTL::Buffer * tmpBuffer = nullptr;
    if (iter == end(freeMatvecTemporaryBuffers)) {
        tmpBuffer = Metal::newBuffer(stripDimGroups * numRows * sizeof(float));
    } else {
        tmpBuffer = iter->second;
        freeMatvecTemporaryBuffers.erase(iter);
    }
    inUseMatvecTemporaryBuffers.emplace(tmpKey, tmpBuffer);

    MTL::CommandBuffer * commandBuffer = Metal::getCommandBuffer(0);
    long rowGroups = numRows / (stripLength * 4);
    Metal::queueCall(commandBuffer, *matvecFunc,
                     stripLength, 1, 1,
                     rowGroups, stripDimGroups, 1,
                     mat, in, tmpBuffer,
                     numRows, numCols);

    Metal::queueCall(commandBuffer, *reducerFunc,
                     stripLength, 1, 1,
                     rowGroups, 1, 1,
                     tmpBuffer, out,
                     numRows, stripDimGroups, outputOffsetElements);

    //Metal::waitUntilCompleted(0, reclaimMatvecBuffers);
    /*
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    commandBuffer->release();
     */
}

void reclaimMatvecBuffers() {
    for (auto iter : inUseMatvecTemporaryBuffers) {
        freeMatvecTemporaryBuffers.emplace(iter.first, iter.second);
    }
    inUseMatvecTemporaryBuffers.clear();
}

Metal::Function * multiheadMatvecFunc = nullptr;
Metal::Function * multiheadReducerFunc = nullptr;

void initMultiheadMatvec() {
    static string matvecSrc = R"(
            #include <metal_stdlib>
            using namespace metal;

            /*
             *  This requires the key matrix has been transposed
             *  groupDim.y needs to divide headDimension
             *
             */
            kernel void singleColumnMatvec(device const bfloat4 * mat [[buffer(0)]],
                                        device const bfloat * vec [[buffer(1)]],
                                        device float4 * result [[buffer(2)]],
                                        constant int64_t & headDimension [[buffer(3)]],
                                        constant int64_t & numHeads [[buffer(4)]],
                                        constant int64_t & numRows [[buffer(5)]],
                                        constant int64_t & leadingDimension [[buffer(6)]],
                                        uint2 threadPos [[thread_position_in_threadgroup]],
                                        uint2 threadDim [[threads_per_threadgroup]],
                                        uint2 groupPos [[threadgroup_position_in_grid]],
                                        uint2 groupDim [[threadgroups_per_grid]] ) {
                int64_t leadingDimension_4 = leadingDimension >> 2;
                int64_t threadRowOffset = threadPos.x + groupPos.x * threadDim.x;
                uint headOffset = 0;
                for (uint head = 0; head < numHeads; ++head) {
                    float4 t = 0;
                    for (uint h = 0; h < headDimension; h+=groupDim.y) {
                        uint hIdx = h + headOffset;
                        t += static_cast<float4>(mat[threadRowOffset + (hIdx+groupPos.y) * leadingDimension_4] *
                            vec[hIdx+groupPos.y]);
                    }
                    headOffset += headDimension;
                    result[threadRowOffset + groupPos.y * leadingDimension_4 + head * groupDim.y * leadingDimension_4] = t;
                }
            }
        )";

    static string reducerSrc = R"(
            #include <metal_stdlib>
            using namespace metal;

            kernel void singleColumnMatvecReducer(
                        device float4 * input [[buffer(0)]],
                        device bfloat4 * result [[buffer(1)]],
                        constant int64_t & numTemporaries [[buffer(2)]],
                        constant int64_t & numHeads [[buffer(3)]],
                        constant int64_t & numRows [[buffer(4)]],
                        constant int64_t & leadingDimension [[buffer(5)]],
                        uint2 threadPos [[thread_position_in_threadgroup]],
                        uint2 threadDim [[threads_per_threadgroup]],
                        uint2 groupPos [[threadgroup_position_in_grid]],
                        uint2 groupDim [[threadgroups_per_grid]] ) {
                int64_t leadingDimension_4 = leadingDimension >> 2;
                int64_t numRows_4 = numRows >> 2;
                int64_t threadRowOffset = threadPos.x + groupPos.x * threadDim.x;
                for (uint head = 0; head < numHeads; ++head) {
                    float4 t = 0;
                    for (uint j = 0; j < numTemporaries; ++j) {
                        t += input[threadRowOffset + j * leadingDimension_4 + head * numTemporaries * leadingDimension_4];
                    }
                    result[threadRowOffset + head * numRows_4] = static_cast<bfloat4>(t);
                }
            }
        )";

    multiheadMatvecFunc = Metal::getFunction(matvecSrc, "singleColumnMatvec");
    multiheadReducerFunc = Metal::getFunction(reducerSrc, "singleColumnMatvecReducer");
}

multimap<pair<int, int>, MTL::Buffer * > inUseMultiheadMatvecBuffers;
multimap<pair<int, int>, MTL::Buffer * > freeMultiheadMatvecBuffers;

void cleanupMultiheadMatvecPass() {
    for (auto & p : inUseMultiheadMatvecBuffers) {
        freeMultiheadMatvecBuffers.emplace(p.first, p.second);
    }
    inUseMultiheadMatvecBuffers.clear();
}

void multiheadMatvecMetal(MTL::Buffer * mat,
                          MTL::Buffer * in,
                          MTL::Buffer * out,
                          long headDimension,
                          long numHeads,
                          long numRows,
                          long leadingDimension) {

    if (!multiheadMatvecFunc) {
        initMultiheadMatvec();
    }

    long stripDimGroups = 8;

    tuple<int, int> tmpKey = make_tuple(numRows, stripDimGroups);
    auto iter = freeMultiheadMatvecBuffers.find(tmpKey);
    MTL::Buffer * tmpBuffer;
    if (iter == end(freeMultiheadMatvecBuffers)) {
        tmpBuffer = Metal::newBuffer(numHeads * stripDimGroups * leadingDimension * sizeof(float));
    } else {
        tmpBuffer = iter->second;
        freeMultiheadMatvecBuffers.erase(iter);
    }
    inUseMultiheadMatvecBuffers.emplace(tmpKey, tmpBuffer);

    long stripLength = 32;

    long rowGroups = numRows / (stripLength * 4);

    MTL::CommandBuffer * commandBuffer = Metal::getCommandBuffer(0);
    Metal::queueCall(commandBuffer, *multiheadMatvecFunc,
                     stripLength, 1, 1,
                     rowGroups, stripDimGroups, 1,
                     mat, in, tmpBuffer,
                     headDimension, numHeads, numRows, leadingDimension);

    Metal::queueCall(commandBuffer, *multiheadReducerFunc,
                     stripLength, 1, 1,
                     rowGroups, 1, 1,
                     tmpBuffer, out,
                     stripDimGroups, numHeads, numRows, leadingDimension);

    Metal::waitUntilCompleted(0, cleanupMultiheadMatvecPass);
}

/*
template<typename T>
void multiplyMatrices(const enum CBLAS_ORDER ORDER,
                      const enum CBLAS_TRANSPOSE TRANSA,
                      const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                      const int K, const T ALPHA, MTL::Buffer * A, const int LDA,
                      MTL::Buffer * B, const int LDB, const T BETA, MTL::Buffer * C,
                      const int LDC) {
    multiplyMatrices(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A->contents(), LDA,
                     B->contents(), LDB, BETA, C->contents(), LDC);
}

*/