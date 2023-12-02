//
// Created by grady on 8/6/22.
//

#ifndef FUNDAMENTALS_PROCESSOR_TIMER_H
#define FUNDAMENTALS_PROCESSOR_TIMER_H

#include<chrono>

using namespace std;

class Timer {
    chrono::time_point<chrono::system_clock> startTime;
public:
    Timer();
    void start();
    double elapsed() const;
    double elapsedSeconds() const;
    long elapsedNanos() const;
};


#endif //FUNDAMENTALS_PROCESSOR_TIMER_H
