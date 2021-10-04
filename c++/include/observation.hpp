#pragma once

#include <cstdint>
#include <vector>

using std::vector;
using byte = uint8_t;

class Observation {
    vector<byte> mFeatures;
    byte mLabel;

  public:
    const vector<byte> &get_features() const noexcept { return mFeatures; }
    byte get_label() const noexcept { return mLabel; }
    Observation(vector<byte> feature_vector, byte label) noexcept
        : mFeatures(feature_vector), mLabel(label) {}
    ~Observation() { /*all members RAII already*/
    }
};
