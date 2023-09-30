//
// Created by Grady Schofield on 9/30/23.
//

#ifndef STREAMING_LLAMA_CHECKER_H
#define STREAMING_LLAMA_CHECKER_H

#include<barrier>
#include<functional>
#include<numeric>
#include<vector>

using namespace std;

class CheckerData {
    vector<float> data;
    vector<int> dimensions;
    int leadingDimension;
public:
    CheckerData(vector<float> && data, vector<int> const & dimensions, int leadingDimension);
    vector<float> const & getData() const;
    vector<int> const & getDimensions() const;
    int getLeadingDimension() const;
};

template<typename T>
unique_ptr<CheckerData> createDataAccessor(T * ptr, vector<int> dimension, int leadingDimension);

class Checker {
    barrier<function<void()>> bar;
    barrier<function<void()>> resetBar;
    float tolerance;
    vector<unique_ptr<CheckerData>> accessors;
    float runningTolerance;

    void check() {
        if(accessors.size() != 2) {
            throw 1;
        }
        CheckerData const * d1 = accessors[0].get();
        CheckerData const * d2 = accessors[1].get();
        if(d1->getDimensions() != d2->getDimensions()) {
            throw 1;
        }
        if(d1->getDimensions().size() != 2) {
            throw 1;
        }
        float absMaxValue = 0;
        for(int j = 0; j < d1->getDimensions()[1]; ++j) {
            float const * p1 = &d1->getData()[j * d1->getLeadingDimension()];
            float const * p2 = &d2->getData()[j * d2->getLeadingDimension()];
            for(int i = 0; i < d1->getDimensions()[0]; ++i) {
                float largestPositive = max(fabs(p1[i]), fabs(p2[i]));
                absMaxValue = max(absMaxValue, largestPositive);
            }
        }
        for(int j = 0; j < d1->getDimensions()[1]; ++j) {
            float const * p1 = &d1->getData()[j * d1->getLeadingDimension()];
            float const * p2 = &d2->getData()[j * d2->getLeadingDimension()];
            for(int i = 0; i < d1->getDimensions()[0]; ++i) {
                if(p1[i] != 0 && p2[i] != 0) {
                    float largestPositive = max(fabs(p1[i]), fabs(p2[i]));
                    float relErr = fabs(p1[i] - p2[i]) / largestPositive;
                    if(largestPositive < absMaxValue * runningTolerance) {
                    } else {
                        if (relErr > runningTolerance) {
                            throw 1;
                        }
                    }
                }
            }
        }
        accessors = vector<unique_ptr<CheckerData>>();
        runningTolerance += tolerance;
    }

    void resetTolerance() {
        runningTolerance = tolerance;
    }

public:
    Checker(float tolerance)
        : bar(2, bind(&Checker::check, this)),
        resetBar(2, bind(&Checker::resetTolerance, this)),
        tolerance(tolerance), runningTolerance(tolerance)
    {
    }

    void submitResult(unique_ptr<CheckerData> && accessor) {
        accessors.push_back(std::move(accessor));
        bar.arrive_and_wait();
    }

    void finish() {
        resetBar.arrive_and_wait();
    }
};


#endif //STREAMING_LLAMA_CHECKER_H
