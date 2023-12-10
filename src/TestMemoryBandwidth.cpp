//
// Created by Grady Schofield on 11/29/23.
//

#include<iostream>
#include<mutex>
#include<thread>
#include<vector>
#include<Timer.h>

using namespace std;

int main(int argc, char ** argv) {
    int numThreads = 4;
    typedef uint16_t UnitType;
    long stride = 4096 * 4096;
    long size = stride * sizeof(UnitType) * ((long)10E9 / (stride*sizeof(UnitType)));
    long len  = size / sizeof(UnitType);
    cout << "stride: " << stride << endl;
    cout << "size: " << size << endl;
    UnitType* p;
    posix_memalign((void**)&p, 64, size);
    memset(p, 0, size);
    mutex m;
    auto worker = [len, stride, numThreads, &m](UnitType * p, int threadIdx) {
        long threadOffset = threadIdx * (stride / numThreads);
        long threadLen = stride / numThreads;
        {
            lock_guard lk(m);
            cout << threadIdx << " " << threadOffset << " " << threadLen << "\n";
        }
        for (int j = 0; j < 10; ++j) {
            long tmp = 0;
            for (long start = threadOffset; start + threadLen < len; start += stride) {
                for (long i = start; i < start + threadLen; ++i) {
                    tmp += p[i];
                }
            }
            cout << tmp << "\n";
        }
    };
    vector<thread> threads;
    Timer timer;
    timer.start();
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker, p, i);
    }
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
    cout << 10 * size / timer.elapsed() / 1E9 << " GB/s\n";
    return 0;
}