//
// Created by Grady Schofield on 12/2/23.
//

#ifndef STREAMING_LLAMA_EVALUATIONTIMINGS_H
#define STREAMING_LLAMA_EVALUATIONTIMINGS_H

#include<cstdint>
#include<list>
#include<map>

using namespace std;

class EvaluationTimings {
    list<char const *> appearanceOrder;
    map<char const *, long> times;
    map<char const *, timespec> currentTime;
public:
    void start(char const * name) {
        auto iter = times.find(name);
        timespec t;
        if (iter == end(times)) {
            appearanceOrder.push_back(name);
            timespec & out = currentTime[name];
            clock_gettime(CLOCK_REALTIME, &t);
            out = t;
        } else {
            timespec & out = currentTime[name];
            clock_gettime(CLOCK_REALTIME, &t);
            out = t;
        }
    }

    void finish(char const * name) {
        timespec endTime;
        clock_gettime(CLOCK_REALTIME, &endTime);
        timespec startTime = currentTime.at(name);
        times[name] += (endTime.tv_sec - startTime.tv_sec) * 1E9L + (endTime.tv_nsec - startTime.tv_nsec);
    }

    void print(ostream & os) {
        long totalTime = 0;
        for (auto & p : times) {
            totalTime += p.second;
        }
        for (char const * s : appearanceOrder) {
            os << s << " " << times[s] / (double)totalTime << endl;
        }
    }
};

#endif //STREAMING_LLAMA_EVALUATIONTIMINGS_H
