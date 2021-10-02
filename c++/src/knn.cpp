#include "knn.hpp"
#include "observation.hpp"
#include "subset.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <cassert>

using std::vector;

Subset<KNN::data_point> KNN::find_k_nearest(data_point& query_point) {
    auto distances = vector<double>();
    distances.reserve(training_dataset.size());

    for (const auto& data : training_dataset) {
	distances.push_back( KNN::get_distance(query_point, data) );
    }

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
	    distance += std::pow( std::to_integer<uint8_t>(query_features[i])
				    - std::to_integer<uint8_t>(point_features[i]), 2 );
	}

	distance = std::sqrt(distance);
    } else {
	// TODO: Add Manhattan distance
    }
}

void KNN::predict() {

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

KNN::KNN(vector<data_point> dataset):
    k(5),	// just to initialize, this will be calculated later
    dataset(dataset)
{
    Subset subset(dataset, {});

    this->split_dataset();
}

