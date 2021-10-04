#pragma once
#include "observation.hpp"
#include "subset.hpp"
#include <vector>
#include <map>
#include <span>

using std::vector;
using std::span;

class KNN {
    using data_point = Observation;
    
    int k;
    vector<data_point> dataset;	// complete dataset
    span<data_point> training_dataset,
		      validation_dataset,
		      testing_dataset;

    private:
	Subset<data_point,std::span<data_point>>
		find_k_nearest(const vector<byte>& query_point_features) const;
	void split_dataset() noexcept;
	static double get_distance(const vector<byte>& query_features,
				   const vector<byte>& point_features) noexcept;

    public:
	void train();
	byte predict(const vector<byte>& feature_vectors) const;	// we don't pass the data_point class here, it already has the label
	std::map<byte,double> predict_with_accuracies(const vector<byte>& feature_vectors) const;	// we don't pass the data_point class here, it already has the label
	double get_test_performance() const;

	KNN(vector<data_point> dataset);
};

