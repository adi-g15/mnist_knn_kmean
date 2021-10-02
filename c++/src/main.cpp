#include "observation.hpp"
#include <cstddef>
#include <cstdint>
#include <ios>
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <bit>
#include "endian.hpp"

using std::vector;
using std::ifstream;
using std::byte;

template<typename T>
T convert_to_native_endian(T n) {
    if constexpr ( std::endian::big /*dataset_endianness*/ != std::endian::native ) {
	// originally our machine is not-big endian, ie. little endian (talking
	// about this planet only), but dataset is big-endian, so swap
	// endianness
	n = endian::swap_endian(n);
    }
    return n;
}

void assert_magic_number_is_ok(ifstream& file, int32_t magic_number) {
    if( !file ) {
	std::cerr << "File can't be opened for reading !\n";
	std::exit(5);	// I/O error
    }

    int32_t n;
    file.read((char*)&n, sizeof(n));

    n = convert_to_native_endian(n);

    if(n != magic_number) {
    	std::cerr << "Magic Number didn't match; Expected: " << magic_number
		  << " Found: " << n << '\n';
	std::exit(1);
    }
}

// Invariant: Expecting the magic numbers (first 4 bytes) have been read
vector<Observation> read_features_labels(ifstream& images, ifstream& labels) {
    auto dataset = vector<Observation>();

    int32_t num_images, num_labels;
    images.read((char*)&num_images, sizeof(num_images));
    labels.read((char*)&num_labels, sizeof(num_labels));
    num_images = convert_to_native_endian(num_images);
    num_labels = convert_to_native_endian(num_labels);

    assert(num_images == num_labels && "Number of images and labels not same");

    int32_t num_rows, num_cols;	// in each image
    images.read((char*)&num_rows, sizeof(num_rows));
    images.read((char*)&num_cols, sizeof(num_cols));
    num_rows = convert_to_native_endian(num_rows);
    num_cols = convert_to_native_endian(num_cols);

    byte label;
    auto pixels = vector<byte>(num_rows*num_cols);

    std::cout << "Going to read " << num_images << std::endl;
    for (auto _i=0; _i < num_images; _i++) {
	labels.read((char*)&label, sizeof(label));
	images.read((char*)pixels.data(), num_rows*num_cols);

	dataset.push_back( Observation(pixels,label) );
    }

    return dataset;
}

int main() {
    auto train_dataset = vector<Observation>();
    auto test_dataset = vector<Observation>();

    switch (std::endian::native) {
	case std::endian::big: std::cout << "My machine is Big endian" << std::endl; break;
	case std::endian::little : std::cout << "My machine is Little endian" << std::endl; break;
	default: std::cout << "This is from another planet" << std::endl; break;
    };

    auto train_images_file =
	ifstream("dataset/train-images-idx3-ubyte", std::ios_base::binary);
    auto train_labels_file =
	ifstream("dataset/train-labels-idx1-ubyte", std::ios_base::binary);
    auto test_images_file =
	ifstream("dataset/t10k-images-idx3-ubyte", std::ios_base::binary);
    auto test_labels_file =
	ifstream("dataset/t10k-labels-idx1-ubyte", std::ios_base::binary);

    assert_magic_number_is_ok(train_images_file, 2051);
    assert_magic_number_is_ok(train_labels_file, 2049);
    assert_magic_number_is_ok(test_images_file, 2051);
    assert_magic_number_is_ok(test_labels_file, 2049);

    std::cout << "Verified all magic numbers!\n";

    train_dataset = read_features_labels(train_images_file, train_labels_file);
    test_dataset  = read_features_labels(test_images_file, test_labels_file);

    std::cout << "Read " << train_dataset.size() << " observations from training data\n";
    std::cout << "Read " << test_dataset.size() << " observations from test data\n";

    // ETL - Extracted, Loaded... tranformed in the sense we created object for each image data
}

