//
// Created by Grady Schofield on 11/29/23.
//

#include<iostream>
#include<thread>
#include<vector>
#include<Timer.h>

using namespace std;

int main(int argc, char ** argv) {
    int numThreads = 4;
    long size = 10E9 / numThreads;
    long len  = size / 8;
    vector<long*> p(numThreads);
    for(int i = 0; i < numThreads; ++i) {
        posix_memalign((void**)&p[i], 64, size);
        memset(p[i], 0, size);
    }
    auto worker = [len](long * p) {
        long tmp = 0;
        for (int i = 0; i < len; ++i) {
            tmp += p[i];
        }
        cout << tmp << "\n";
    };
    vector<thread> threads;
    Timer timer;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker, p[i]);
    }
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
    cout << numThreads * size / timer.elapsed() / 1E9 << " GB/s\n";
    return 0;
}