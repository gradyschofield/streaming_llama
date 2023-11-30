//
// Created by Grady Schofield on 11/29/23.
//

#include<barrier>
#include<iostream>
#include<thread>
#include<vector>
#include<Timer.h>

#ifdef __APPLE__
#include<Accelerate/Accelerate.h>
#endif

using namespace std;

int main(int argc, char ** argv) {
    int numThreads = 4;
    //long M = numThreads * 8 * (29580/(numThreads * 8));
    long M = numThreads * 8 * (4096/(numThreads * 8));
    long K = M;
    if (K % numThreads != 0 || K % 8 != 0) {
        cout << "Num threads must divide num columns\n";
        return 1;
    }
    long matrixSize = M * K * 4;
    long numMatrices = 14E9 / matrixSize;
    vector<float*> A(numMatrices);
    vector<float*> x(numMatrices);
    vector<float*> y(numMatrices);
    for(int i = 0; i < numMatrices; ++i) {
        posix_memalign((void**)&A[i], 64, M * K * 4);
        posix_memalign((void**)&x[i], 64, K * 4);
        posix_memalign((void**)&y[i], 64, M * 4);
        memset(A[i], 0, M * K * 4);
        memset(x[i], 0, K*4);
    }
    vector<float*> localAccumulator(numThreads);
    vector<atomic<int>> ci(numThreads);
    for(int i = 0; i < numThreads; ++i) {
        posix_memalign((void **) &localAccumulator[i], 64, M * 4);
    }
    auto accumulateLocal = [M, numMatrices, numThreads, &localAccumulator, &y, &ci]() {
        static int matrixIdx = 0;
        for (int i = 0; i < numThreads; ++i) {
            atomic_load_explicit(&ci[i], memory_order_relaxed);
        }
        atomic_thread_fence(memory_order_acquire);
        float * out = y[matrixIdx];
        memset(out, 0, M * 4);
        for (int i = 0; i < numThreads; ++i) {
            float * tmp = localAccumulator[i];
            for (int row = 0; row < M; ++row) {
                out[row] += tmp[row];
            }
        }
        matrixIdx = (matrixIdx + 1) % numMatrices;
    };
    barrier bar(numThreads, accumulateLocal);
    Timer timer;
    auto worker = [M, K, numThreads, numMatrices, &A, &x, &localAccumulator, &bar, &ci](int threadIdx) {
        int startColumn = threadIdx * (K / numThreads);
        int endColumn = (threadIdx+1) * (K / numThreads);
        float * tmp = localAccumulator[threadIdx];
        for (int i = 0; i < numMatrices; ++i) {
            memset(tmp, 0, M*4);
            float * mat = A[i];
            float * _x = x[i];
            float * colp = &mat[startColumn * M];
            long start = startColumn * M;
            long end = endColumn * M;
            for (int col = startColumn; col < endColumn; col += 1) {
                float * col1 = &mat[col*M];
                for (int row = 0; row < M; ++row) {
                    tmp[row] += col1[row] * _x[col];
                }
            }
            atomic_thread_fence(memory_order_release);
            atomic_store_explicit(&ci[threadIdx], 1, memory_order_relaxed);
            bar.arrive_and_wait();
        }
    };
    vector<thread> threads;
    timer.start();
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker, i);
    }
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
    cout << numMatrices * matrixSize / timer.elapsed() / 1E9 << " GB/s\n";
    timer.start();
    for (int i = 0; i < numMatrices; ++i) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, 1, K, 1.0, A[i], M, x[i], K, 0.0, y[i], M);
    }
    cout << numMatrices * matrixSize / timer.elapsed() / 1E9 << " GB/s\n";
    return 0;
}
