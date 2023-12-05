//
// Created by Grady Schofield on 11/29/23.
//

#include<iostream>
#include<thread>
#include<vector>
#include<Timer.h>

using namespace std;

int main(int argc, char ** argv) {
    int numThreads = 8;
    long size = 10E9 / numThreads;
    typedef uint16_t UnitType;
    long len  = size / sizeof(UnitType);
    vector<UnitType*> p(numThreads);
    for(int i = 0; i < numThreads; ++i) {
        posix_memalign((void**)&p[i], 64, size);
        memset(p[i], 0, size);
    }
    auto worker = [len](UnitType * p) {
        for (int j = 0; j < 10; ++j) {
            long tmp = 0;
            for (int i = 0; i < len; ++i) {
                tmp += p[i];
            }
            cout << tmp << "\n";
        }
    };
    vector<thread> threads;
    Timer timer;
    timer.start();
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker, p[i]);
    }
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
    cout << 10 * numThreads * size / timer.elapsed() / 1E9 << " GB/s\n";
    return 0;
}