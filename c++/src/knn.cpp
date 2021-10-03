#include "knn.hpp"
#include "observation.hpp"
#include "subset.hpp"
#include <algorithm>
#include <array>
#include <bits/ranges_algo.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <pstl/glue_execution_defs.h>
#include <random>
#include <cassert>
#include <span>
#include <utility>
#include <vector>
#include <execution>

using std::vector;

// Invariant: k is less than or equal to training dataset size
Subset<KNN::data_point,std::span<KNN::data_point>> KNN::find_k_nearest(const vector<byte>& query_point_features) {
    auto distances = vector<std::pair<double,size_t>>();
    distances.reserve(training_dataset.size());

    auto nearest_elements_indices = vector<uint64_t>();	// instead of storing copy of objects, we use the memory efficient Subset class

    for (auto i = 0; i < training_dataset.size(); ++i) {
	distances.push_back({
	    KNN::get_distance(query_point_features, training_dataset[i].get_features()), i
	    });
    }

    // Fill in "nearest_elements_indices" vector, this is NlogN currently
    // TODO: improve it to atleast NK^2
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

double KNN::get_distance(const vector<byte>& query_features,
			        const vector<byte>& point_features) noexcept {
    constexpr bool USE_EUCLIDEAN_DISTANCE = true;

    if constexpr ( USE_EUCLIDEAN_DISTANCE ) {
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

byte KNN::predict(const vector<byte>& feature_vector) {
    auto k_neighbours = find_k_nearest(feature_vector);

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

void KNN::split_dataset() noexcept {
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

    std::cout << "Lengths of datasets: \n\tTraining: " << training_dataset.size()
		<< "\n\tValidation: " << validation_dataset.size()
		<< "\n\tTesting: " << testing_dataset.size() << '\n';
}

void KNN::train() {	// using validation_dataset to chose good 'k'
    auto max_correct_count = 0;
    auto best_k = 1;
    // Just trying k=0 to k=5 for now
    for(auto i=1; i<5; ++i) {
	auto count_correct = 0, j = 1;
	std::cout << "Validating for k=" << i << '\n';
	for(const auto& data: validation_dataset) {
	    auto predicted_label = this->predict(data.get_features());

	    if( predicted_label == data.get_label() ) {
		++count_correct;
	    }/* else {
		std::cout << "[Wrong] ";
	    }*/

	    if( j % 1024 == 0 )
		std::cout << "\tEpochs: " << j << "; Current accuracy = " << (count_correct*100.0f)/j << '%' << std::endl;
	    // std::cout << int(data.get_label()) << " -> " << int(predicted_label) << "; current accuracy = " << (count_correct*100.0f)/j << '%' << std::endl;
	    ++j;
	}

	if ( count_correct > max_correct_count ) {
	    best_k = i;
	}
    }

    std::cout << "Optimal value of k after validating: " << best_k << '\n';
    this->k = best_k;
}

double KNN::get_test_performance() {
    auto count_correct = 
	std::count_if(
		std::execution::par_unseq,
		testing_dataset.begin(), training_dataset.end(),
		[this](const auto& data) {
		    return predict(data.get_features()) == data.get_label();
		});

    return (double)(count_correct) / testing_dataset.size();
}

KNN::KNN(vector<data_point> dataset):
    k(5),	// just to initialize, this will be calculated later
    dataset(dataset)
{
    this->split_dataset();
}

