//
// Created by grady on 8/6/22.
//

#include "Timer.h"

Timer::Timer()
    : startTime(chrono::system_clock::now())
{
}

void Timer::start() {
    startTime = chrono::system_clock::now();
}

double Timer::elapsed() const {
    auto endTime = chrono::system_clock::now();
    return chrono::duration_cast<chrono::nanoseconds>(endTime - startTime).count() / 1E9;
}
