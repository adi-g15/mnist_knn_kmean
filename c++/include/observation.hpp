#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

using std::vector;
using std::byte;

class Observation {
    vector<byte> features;
    byte label;

    public:
    Observation(vector<byte> feature_vector, byte label) noexcept:
	features(feature_vector),
	label(label) {}
    ~Observation() { /*all members RAII already*/ }
};

