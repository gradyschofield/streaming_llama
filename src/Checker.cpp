//
// Created by Grady Schofield on 9/30/23.
//

#include<Bf16.h>
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
    int cdrProduct = accumulate(begin(dimension)+1, end(dimension), 1, std::multiplies<int>());
    vector<float> data(leadingDimension * cdrProduct);
    memcpy(data.data(), ptr, data.size());
    return make_unique<CheckerData>(std::move(data), dimension, leadingDimension);
}

template<>
unique_ptr<CheckerData> createDataAccessor(Bf16 * ptr, vector<int> dimension, int leadingDimension) {
    int cdrProduct = accumulate(begin(dimension)+1, end(dimension), 1, std::multiplies<int>());
    vector<float> data(leadingDimension * cdrProduct);
    for(int i = 0; i < leadingDimension * cdrProduct; ++i) {
        data[i] = ptr[i].toFloat();
    }
    return make_unique<CheckerData>(std::move(data), dimension, leadingDimension);
}

