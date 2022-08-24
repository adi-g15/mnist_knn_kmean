#include "kmean.hpp"
#include <cstdlib>
#include <vector>

using std::vector;

KMeans::KMeans(vector<KMeans::data_point> dataset) {}

uint32_t KMeans::find_optimal_k() const {
    // TODO
}

void KMeans::train(uint32_t num_clusters) {
    if( num_clusters == 0 ) {
	num_clusters = this->find_optimal_k();
    }

    clusters.clear();
    clusters.reserve(num_clusters);

    for (auto i=0; i < num_clusters; ++i) {
	// Randomly chose a point and add it to a new cluster
	const auto idx = rand() % training_data.size();
	clusters.push_back( Cluster(
		training_data,
		idx
	));
    }
}
