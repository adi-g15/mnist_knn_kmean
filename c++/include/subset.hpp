#pragma once

#include <algorithm>
#include <vector>

using std::vector;

// Currently only for vector
template<typename T>
class Subset {
    // Idea from the "Competetive programmer handbook", we can use a binary
    // number to denote a subset, with 1 for indices of elements included in
    // this 'subset' and 0 for indices not in this subset
    const vector<T>& _original_set;
    vector<bool> _subset;
    int length;

    public:
	Subset(const vector<T>& original_set, vector<bool> subset):
	    _original_set(original_set),
	    _subset(subset) {

	    length = std::count(_subset.begin(), _subset.end(), true);
	}
};

