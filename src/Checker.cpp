//
// Created by Grady Schofield on 9/30/23.
//

#include "Checker.h"

CheckerData::CheckerData(vector<float> && data,
                         vector<int> const & dimensions,
                         int leadingDimension)
    : data(std::move(data)), dimensions(dimensions), leadingDimension(leadingDimension)
{
}

vector<float> const & CheckerData::getData() const {
    return data;
}

vector<int> const & CheckerData::getDimensions() const {
    return dimensions;
}

int CheckerData::getLeadingDimension() const {
    return leadingDimension;
}

template<>
unique_ptr<CheckerData> createDataAccessor(float * ptr, vector<int> dimension, int leadingDimension) {
    float * fptr = (float*)ptr;
    int cdrProduct = accumulate(begin(dimension)+1, end(dimension), 1, std::multiplies<int>());
    vector<float> data(leadingDimension * cdrProduct);
    memcpy(data.data(), fptr, data.size());
    return make_unique<CheckerData>(std::move(data), dimension, leadingDimension);
}

