#pragma once

#include "observation.hpp"
#include "subset.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

using byte = uint8_t;

// 'idx' signifies 'index' wherever used in this whole codebase
class KMeans {
    using data_point = Observation;

    std::vector<data_point> dataset;
    std::span<data_point> training_data, validation_data, testing_data;

    class Cluster {
	byte most_frequent_label;
        Subset<std::span<data_point>> members;

	vector<double> centroid;
	public:
	Cluster(const std::span<data_point>& dataset, size_t idx_initial_data_point): members(dataset) {
	    members.insert_index(idx_initial_data_point);

	    const auto& dp = dataset[idx_initial_data_point];
	    most_frequent_label = dp.get_label();

	    centroid.reserve( dp.get_features().size() );
	    for(const auto& b: dp.get_features() ) {
		centroid.push_back( b );
	    }
	}
    };

    std::vector<Cluster> clusters; // k = clusters.size()
    // int k;

  public:
    uint32_t find_optimal_k() const;
    void train(uint32_t num_clusters = 0);
    void predict(std::vector<byte> &feature_vector);

    KMeans(vector<data_point> complete_dataset);
};
