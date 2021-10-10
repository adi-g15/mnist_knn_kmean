#include "knn.hpp"
#include "observation.hpp"
#include "subset.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

using std::vector;

// https://www.geeksforgeeks.org/k-largestor-smallest-elements-in-an-array/
vector<size_t> get_k_smallest_indices(const vector<double> &v, int k) {
    // we could sort, then chose k smallest but that is NlogN, we want even
    // better, ie. O(Nk), k is generally small, so it's better

    vector<size_t> idx(k);
    std::iota(idx.begin(), idx.end(), 0); // fill idx with 0,1,2,3,4...k

    // O(Nk)... k is generally small, there are faster methods at link given
    // above
    // NOTE: idx is an array of vectors, so actual value is received by using
    // v[ idx[i] ], also the current_max_idx is same, it is INDEX IN idx ARRAY
    // NOT v, so it stores the 'index which contains the index to the value in
    // v'
    for (auto i = 0; i < v.size(); ++i) {
        auto current_max_idx = 0;
        for (auto j = 1; j < k; ++j) {
            if (v[idx[j]] > v[idx[current_max_idx]]) {
                current_max_idx = j;
            }
        }

        if (v[i] < v[idx[current_max_idx]]) {
            idx[current_max_idx] = i;
        }
    }

    return idx;
}

// Invariant: k is less than or equal to training dataset size
Subset<span<KNN::data_point>>
KNN::find_k_nearest(const vector<byte> &query_point_features) const {
    auto distances = vector<double>();
    distances.reserve(training_dataset.size());

    for (auto i = 0; i < training_dataset.size(); ++i) {
        distances.push_back(KNN::get_distance(
            query_point_features, training_dataset[i].get_features()));
    }

    // Fill in "nearest_elements_indices" vector, this is NlogN currently
    const auto k_smallest_distances_idx =
        get_k_smallest_indices(distances, k); // get 7 smallest

    // instead of storing copy of objects, we use the memory efficient Subset
    // class this is just bit trickery, for eg. Let original set be {1,2,3,4,5},
    // consider these ways to store the subset {1,2}:
    // 1. Create another set, add 1 and 2, ... so we get {1,2}, but for large
    // objects this will require copying those objects
    // 2. Bit Manipulation: The same subset can be represented as "11000", WITH
    // A REFERENCE to original set, so we know 1st and 2nd are included in this
    // subset
    auto subset = Subset<span<data_point>>(training_dataset);

    for (auto index : k_smallest_distances_idx) {
        subset.insert_index(index);
    }

    return subset;
}

double KNN::get_distance(const vector<byte> &query_features,
                         const vector<byte> &point_features) noexcept {
    constexpr bool USE_EUCLIDEAN_DISTANCE = true;

    if constexpr (USE_EUCLIDEAN_DISTANCE) {
        auto distance = 0.0;

        assert(query_features.size() == point_features.size() &&
               "Feature vectors of each point must be of same length");

        for (auto i = 0; i < query_features.size(); ++i) {
            distance += std::pow(query_features[i] - point_features[i], 2);
        }

        distance = std::sqrt(distance);
        return distance;
    } else {
        // TODO: Add Manhattan distance
    }
}

// Returns all labels, and how probable it is to be one of them
std::map<byte, double>
KNN::predict_with_accuracies(const vector<byte> &feature_vector) const {
    auto k_neighbours = find_k_nearest(feature_vector);

    auto label_freq = std::array<int, 256>();
    for (const auto &neighbour : k_neighbours) {
        ++label_freq[neighbour.get_label()];
    }

    // NOTE: MNIST dataset has 10 labels, using that
    auto freq_map = std::map<byte, double>();
    for (auto i = 0; i < 10; ++i) {
        freq_map[i] = double(label_freq[i] * 100) / k_neighbours.size();
    }

    return freq_map;
}

byte KNN::predict(const vector<byte> &feature_vector) const {
    auto k_neighbours = find_k_nearest(feature_vector);

    // std::map<byte,int> label_freq;	// frequencies
    auto label_freq = std::array<int, 256>(); // max 256 1-byte labels possible

    // IMP NOTE: assuming labels start from 0
    for (const auto &neighbour : k_neighbours) {
        ++label_freq[neighbour.get_label()];
    }

    return std::max_element(label_freq.begin(), label_freq.end()) -
           label_freq.begin();
}

void KNN::split_dataset() noexcept {
    std::random_device rd;
    std::mt19937 generator(rd());

    std::shuffle(dataset.begin(), dataset.end(), generator);

    constexpr auto TRAIN_DATA_PERCENTAGE = 0.75;
    constexpr auto VALIDATION_DATA_PERCENTAGE = 0.10;
    constexpr auto TEST_DATA_PERCENTAGE = 0.15;

    const auto num_train_set_elements = TRAIN_DATA_PERCENTAGE * dataset.size();
    const auto num_valid_set_elements =
        VALIDATION_DATA_PERCENTAGE * dataset.size();
    const auto num_test_set_elements = TEST_DATA_PERCENTAGE * dataset.size();

    this->training_dataset = span(dataset.begin(), num_train_set_elements);
    this->validation_dataset =
        span(dataset.begin() + num_train_set_elements, num_valid_set_elements);
    this->testing_dataset =
        span(dataset.begin() + num_train_set_elements + num_valid_set_elements,
             dataset.end());

    std::cout << "Lengths of datasets: \n\tTraining: "
              << training_dataset.size()
              << "\n\tValidation: " << validation_dataset.size()
              << "\n\tTesting: " << testing_dataset.size() << '\n';
}

void KNN::train() { // using validation_dataset to chose good 'k'
    auto max_correct_count = 0;
    auto best_k = 1;
    // Just trying k=0 to k=5 for now
    for (k = 1; k < 5; ++k) {
        // this is basically 'int', using atomic, since we are going to
        // parallelize it
        std::atomic_uint64_t count_correct{0}, i{1};
        std::cout << "Validating for k=" << k << '\n';
        std::for_each(
            std::execution::par_unseq, validation_dataset.begin(),
            validation_dataset.end(),
            [this, &i, &count_correct](const auto &data) {
                auto predicted_label = this->predict(data.get_features());

                if (predicted_label == data.get_label()) {
                    ++count_correct;
                } /* else {
                     std::cout << "[Wrong] ";
                 }*/

                if (i % 1000 == 0)
                    std::cout << "\tEpochs: " << i << "; Current accuracy = "
                              << (double)(count_correct * 100) / i << "%\n";
                // std::cout << int(data.get_label()) << " -> " <<
                // int(predicted_label);
                ++i;
            });

        if (count_correct > max_correct_count) {
            best_k = k;
            max_correct_count = count_correct;
        }
        std::cout << "\tAccuracy: "
                  << (double)(count_correct.load() * 100) /
                         validation_dataset.size()
                  << "\nBest k=" << best_k << '\n';
    }

    std::cout << "Optimal value of k after validating: " << best_k << '\n';
    this->k = best_k;
}

double KNN::get_test_performance() const {
    auto count_correct = std::count_if(
        std::execution::par_unseq, testing_dataset.begin(),
        testing_dataset.end(), [this](const auto &data) {
            return predict(data.get_features()) == data.get_label();
        });

    return (double)(count_correct * 100) / testing_dataset.size();
}

KNN::KNN(vector<data_point> dataset)
    : k(5), // just to initialize, this will be calculated later
      dataset(dataset) {
    this->split_dataset();
}
