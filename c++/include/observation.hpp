#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

using std::vector;
using std::byte;

class Observation {
    vector<byte> mFeatures;
    byte mLabel;

    public:
    const vector<byte>& get_features() const noexcept { return mFeatures; }
    byte get_label() const noexcept { return mLabel; }
    Observation(vector<byte> feature_vector, byte label) noexcept:
	mFeatures(feature_vector),
	mLabel(label) {}
    ~Observation() { /*all members RAII already*/ }
};

