#include "knn.hpp"
#include "observation.hpp"
#include "subset.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <cassert>
#include <span>
#include <utility>
#include <vector>

using std::vector;

// Invariant: k is less than or equal to training dataset size
Subset<KNN::data_point,std::span<KNN::data_point>> KNN::find_k_nearest(const data_point& query_point) {
    auto distances = vector<std::pair<double,size_t>>();
    distances.reserve(training_dataset.size());

    auto nearest_elements_indices = vector<uint64_t>();	// instead of storing copy of objects, we use the memory efficient Subset class

    for (auto i = 0; i < training_dataset.size(); ++i) {
	distances.push_back( { KNN::get_distance(query_point, training_dataset[i]), i} );
    }

    // TODO: Fill in "nearest_elements_indices" vector, this is NlogN currently
    // improve it to atleast NK^2
    std::ranges::sort( distances );
    for(int i=0; i<7; ++i) {
	nearest_elements_indices.push_back(distances[i].second);
    }

    // this is just bit trickery, for eg.
    // Let original set be {1,2,3,4,5}, consider these ways to store the subset
    // {1,2}:
    // 1. Create another set, add 1 and 2, ... so we get {1,2}, but for large
    // objects this will require copying those objects
    // 2. Bit Manipulation: The same subset can be represented as "11000", WITH
    // A REFERENCE to original set, so we know 1st and 2nd are included in this
    // subset
    auto _subset_repr = vector<bool>( training_dataset.size() );

    for (auto index : nearest_elements_indices) {
	_subset_repr[index] = true;
    }

    return Subset<data_point,span<data_point>>(training_dataset, _subset_repr);
}

double KNN::get_distance(const data_point& query_point,
			        const data_point& to) noexcept {
    constexpr bool USE_EUCLIDEAN_DISTANCE = true;

    if constexpr ( USE_EUCLIDEAN_DISTANCE ) {
	const auto &query_features = query_point.get_features();
	const auto &point_features = to.get_features();

	auto distance = 0.0;

	assert(query_features.size() == point_features.size() &&
		"Feature vectors of each point must be of same length");

	for (auto i=0; i<query_features.size(); ++i) {
	    distance += std::pow( query_features[i] - point_features[i], 2 );
	}

	distance = std::sqrt(distance);
	return distance;
    } else {
	// TODO: Add Manhattan distance
    }
}

byte KNN::predict_known(const KNN::data_point& data) {
    auto k_neighbours = find_k_nearest(data);

    // Can use a 256 length array, but for that use uint8_t instead of
    // std::byte, since to use it as index it will ask
    //std::map<byte,int> label_freq;	// frequencies
    auto label_freq = std::array<int,256>();	// max 256 1-byte labels possible

    // IMP NOTE: assuming labels start from 0
    for (const auto& neighbour : k_neighbours) {
	++label_freq[ neighbour.get_label() ];
    }

    assert(!k_neighbours.empty());
    return std::max_element(label_freq.begin(), label_freq.end()) - label_freq.begin();
}

byte KNN::predict(const vector<byte>& feature_vector) {
    // TODO
}

void KNN::split_dataset() {
    std::random_device rd;
    std::mt19937 generator(rd());

    std::shuffle(dataset.begin(), dataset.end(), generator);

    constexpr auto TRAIN_DATA_PERCENTAGE = 0.75;
    constexpr auto VALIDATION_DATA_PERCENTAGE = 0.10;
    constexpr auto TEST_DATA_PERCENTAGE = 0.15;

    const auto num_train_set_elements =
	TRAIN_DATA_PERCENTAGE * dataset.size();
    const auto num_valid_set_elements = 
	VALIDATION_DATA_PERCENTAGE * dataset.size();
    const auto num_test_set_elements = 
	TEST_DATA_PERCENTAGE * dataset.size();

    this->training_dataset =
	span(dataset.begin(), num_train_set_elements);
    this->validation_dataset =
	span(dataset.begin() + num_train_set_elements, num_valid_set_elements);
    this->testing_dataset =
	span(
	    dataset.begin() + num_train_set_elements + num_valid_set_elements,
	    dataset.end()
	    );

    std::cout << "Lengths: \n\t" << training_dataset.size()
		<< "\n\t" << validation_dataset.size()
		<< "\n\t" << testing_dataset.size();
}

void KNN::train() {	// using validation_dataset to chose good 'k'
    // Just trying k=0 to k=5 for now
    for(k=1; k<5; ++k) {
	auto count_correct = 0, i = 1;
	for(const auto& data: validation_dataset) {
	    auto predicted_label = this->predict_known(data);

	    if( predicted_label == data.get_label() ) {
		++count_correct;
	    } else {
		std::cout << "[Wrong] ";
	    }

	    std::cout << int(data.get_label()) << " -> " << int(predicted_label) << "; current accuracy = " << (count_correct*100.0f)/i << '%' << std::endl;
	    ++i;
	}
    }
    // TODO
}

double KNN::get_test_performance() {
    // TODO
}

KNN::KNN(vector<data_point> dataset):
    k(5),	// just to initialize, this will be calculated later
    dataset(dataset)
{
    this->split_dataset();
}

