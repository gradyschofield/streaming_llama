//
// Created by Grady Schofield on 5/10/23.
//

#include<cstdlib>
#include<iostream>
#include<thread>
#include<vector>

#include<Accelerate/Accelerate.h>

#include<Timer.h>

using namespace std;

int main(int argc, char **argv) {
    typedef float fptype;
    vector<thread> threads;
    int M = 256;
    int K = 256;
    int N = 256;
    int numThreads = 1;
    vector<fptype*> a(numThreads);
    vector<fptype*> b(numThreads);
    vector<fptype*> c(numThreads);
    for(int i = 0; i < numThreads; ++i) {
        posix_memalign((void **) &a[i], 256, M * K * sizeof(fptype));
        posix_memalign((void **) &b[i], 256, K * N * sizeof(fptype));
        posix_memalign((void **) &c[i], 256, M * N * sizeof(fptype));
    }
    int nLoops = 20;
    auto doWork = [M,N,K,nLoops](fptype * a, fptype * b, fptype * c) {
        for (int i = 0; i < nLoops; ++i) {
                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            M, N, K, 1.0, a, K, b, K, 0.0, c, M);
                //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                //            M, N, K, 1.0, a, K, b, K, 0.0, c, M);
        }
    };
    Timer timer;
    for(int i = 0; i < numThreads; ++i ) {
        threads.emplace_back(doWork, a[i], b[i], c[i]);
    }
    for(int i = 0; i < numThreads; ++i ) {
        threads[i].join();
    }
    double elapsed = timer.elapsed();
    cout << "GFlops " << 2L * M * N * K / elapsed / 1E9 * nLoops * numThreads<< "\n";
    cout << "sgemm/sec " << nLoops * numThreads / elapsed<< "\n";
}
