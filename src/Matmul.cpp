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
    if (N > 1) {
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
        leftInput.insert(make_tuple(aNumRows, aNumCols));
        rightInput.insert(make_tuple(K, N));
        output.insert(make_tuple(M, N));
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
        leftInput.insert(make_tuple(M, K));
        rightInput.insert(make_tuple(K, N));
        output.insert(make_tuple(M, N));
        static Matvec * matvec = nullptr;
        int numThreads = 8;
        if (!matvec) {
            matvec = new Matvec(numThreads);
        }
        matvec->run(M, N, K, TRANSA, A, LDA, B, C);
        leftInput.insert(make_tuple(M, K));
        rightInput.insert(make_tuple(K, N));
        output.insert(make_tuple(M, N));
        MetalHelpers::getBuffer(A);
        MetalHelpers::getBuffer(B);
        MetalHelpers::getBuffer(C);
        ++numCalls;
        if (numCalls % 10000 == 0) {
            cout << "A inputs: \n";
            for( auto a : leftInput) {
                cout << get<0>(a) << " x " << get<1>(a) << "\n";
            }
            cout << "B inputs: \n";
            for( auto a : rightInput) {
                cout << get<0>(a) << " x " << get<1>(a) << "\n";
            }
            cout << "C outputs: \n";
            for( auto a : output) {
                cout << get<0>(a) << " x " << get<1>(a) << "\n";
            }
        }
    }
}
