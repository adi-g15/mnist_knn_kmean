#include "kmean.hpp"
#include "knn.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <endian.hpp>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

using std::byte;
using std::ifstream;
using std::vector;

template <typename T> T convert_to_native_endian(T n) {
    if constexpr (std::endian::big /*dataset_endianness*/ !=
                  std::endian::native) {
        // originally our machine is not-big endian, ie. little endian (talking
        // about this planet only), but dataset is big-endian, so swap
        // endianness
        n = endian::swap_endian(n);
    }
    return n;
}

void assert_magic_number_is_ok(ifstream &file, int32_t magic_number) {
    if (!file) {
        std::cerr << "File can't be opened for reading !\n";
        std::exit(5); // I/O error
    }

    int32_t n;
    file.read((char *)&n, sizeof(n));

    n = convert_to_native_endian(n);

    if (n != magic_number) {
        std::cerr << "Magic Number didn't match; Expected: " << magic_number
                  << " Found: " << n << '\n';
        std::exit(1);
    }
}

// Invariant: Expecting the magic numbers (first 4 bytes) have been read
vector<Observation> read_features_labels(ifstream &images, ifstream &labels) {
    auto dataset = vector<Observation>();

    int32_t num_images, num_labels;
    images.read((char *)&num_images, sizeof(num_images));
    labels.read((char *)&num_labels, sizeof(num_labels));
    num_images = convert_to_native_endian(num_images);
    num_labels = convert_to_native_endian(num_labels);

    assert(num_images == num_labels && "Number of images and labels not same");

    int32_t num_rows, num_cols; // in each image
    images.read((char *)&num_rows, sizeof(num_rows));
    images.read((char *)&num_cols, sizeof(num_cols));
    num_rows = convert_to_native_endian(num_rows);
    num_cols = convert_to_native_endian(num_cols);

    byte label;
    auto pixels = vector<byte>(num_rows * num_cols);

    std::cout << "Going to read " << num_images << std::endl;
    for (auto _i = 0; _i < num_images; _i++) {
        labels.read((char *)&label, sizeof(label));
        images.read((char *)pixels.data(), num_rows * num_cols);

        dataset.push_back(Observation(pixels, label));
    }

    return dataset;
}

int main() {
    auto train_dataset = vector<Observation>();
    auto test_dataset = vector<Observation>();

    switch (std::endian::native) {
    case std::endian::big:
        std::cout << "My machine is Big endian" << std::endl;
        break;
    case std::endian::little:
        std::cout << "My machine is Little endian" << std::endl;
        break;
    default:
        std::cout << "This is from another planet" << std::endl;
        break;
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
    test_dataset = read_features_labels(test_images_file, test_labels_file);

    std::cout << "Read " << train_dataset.size()
              << " observations from training data\n";
    std::cout << "Read " << test_dataset.size()
              << " observations from test data\n";

    // ETL - Extracted, Loaded... tranformed in the sense we created object for
    // each image data

    // Now create the KNN Model
    auto complete_dataset = std::move(train_dataset);
    complete_dataset.insert(complete_dataset.end(), test_dataset.begin(),
                            test_dataset.end());

    /* K NEAREST NEIGHBOURS (KNN) */
    auto knn_model = KNN(complete_dataset);
    knn_model.train();

    // std::cout << "Testing Accuracy of KNN model: "
    //          << knn_model.get_test_performance() << "%\n";

    // NOTE: Most of the time, we will see one 100%, and all other 0%
    // possibilities, ie. accurate prediction, this is because, here it is among
    // k neighbours only, ie. k is small, if all 4(k) neighbours are 6, then we
    // get 100% chance of 6, which is very probable, since value of k is very
    // low
    const auto &rand_data_point = test_dataset[rand() % test_dataset.size()];
    std::cout << "Taking a random data_point from test_set: Digit "
              << (int)rand_data_point.get_label() << '\n';

    auto possibilities =
        knn_model.predict_with_accuracies(rand_data_point.get_features());
    for (auto &p : possibilities) {
        std::cout << static_cast<int>(p.first) << " -> " << p.second << "%\n";
    }

    // knn_model.predict(data_point) ... we don't have a data point, ie. 28*28

    /* K MEANS (KMEANS) */
    auto kmeans_model = KMeans(complete_dataset);
    kmeans_model.train();
    // byte array to predict now :(
}
