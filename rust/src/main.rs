mod observation;
mod knn;
mod subset;

use observation::Observation;
use knn::KNN;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;  // trait required for read functions on file

fn check_if_magic_matches(bytes: &[u8;4], magic_number: i32) -> Result<(),String> {
    let read_number: i32 = i32::from_be_bytes(*bytes);
    if magic_number == read_number {
        Ok(())
    } else {
        Err(format!("Magic number doesn't match: Expected={}, Read={}", magic_number, read_number))
    }
}

fn main() {
    // Note that these datasets will just be merged and chosing training and
    // testing set is upto the algorithm, not using the MNIST files ratio (6/1)
    let mut train_dataset: Vec<Observation> = Vec::new();
    let mut test_dataset: Vec<Observation> = Vec::new();

    /* Validation set is what we use to validate that our model is trained to
     * our liking, for eg. say when training a neural network we will feed
     * forward and back prop as many times as we need until the performace on
     * the validation set reaches a threshold, then we'll put the test data into
     * the network and see what the performance is*/
    // let mut validation_dataset = Vec::new();

    // Open all 4 dataset files
    // Using Buffered Reader, from doc of io::BufReader:
    // BufReader<R> can improve the speed of programs that make small and 
    // repeated read calls to the same file or network socket. It doesn't help
    // when reading very large amounts at once, or reading a few times.
    // Statistics: In case of this dataset, it saved 0.035 seconds on average
    let mut training_set_label_file = 
        BufReader::new(File::open("dataset/train-labels-idx1-ubyte").unwrap());
    let mut training_set_image_file = 
        BufReader::new(File::open("dataset/train-images-idx3-ubyte").unwrap());
    let mut test_set_label_file = 
        BufReader::new(File::open("dataset/t10k-labels-idx1-ubyte").unwrap());
    let mut test_set_image_file = 
        BufReader::new(File::open("dataset/t10k-images-idx3-ubyte").unwrap());

    // All these check_if_magic_matches, are to make sure we are reading with correct endianness,
    // since the bytes in MNIST dataset are in MSB first (high endian), so on processors such as
    // Intel, these bytes must be reversed since Intel uses low endian
    // ...So... On big endian this is a no-op. On little endian the bytes are swapped.
    let mut four_bytes = [0u8; 4];
    training_set_label_file.read_exact(&mut four_bytes).unwrap();
    check_if_magic_matches(&four_bytes, 2049).unwrap();
    training_set_image_file.read_exact(&mut four_bytes).unwrap();
    check_if_magic_matches(&four_bytes, 2051).unwrap();

    let mut four_bytes = [0u8; 4];
    test_set_label_file.read_exact(&mut four_bytes).unwrap();
    check_if_magic_matches(&four_bytes, 2049).unwrap();
    test_set_image_file.read_exact(&mut four_bytes).unwrap();
    check_if_magic_matches(&four_bytes, 2051).unwrap();

    println!("Verified all magic number correctly !");

    // Both label and image files has num_images, so making sure both are same, since we will be
    // iterating with a pair of files in same loop
    training_set_image_file.read_exact(&mut four_bytes).unwrap();
    let num_images_train = i32::from_be_bytes(four_bytes);
    training_set_label_file.read_exact(&mut four_bytes).unwrap();
    let num_labels_train = i32::from_be_bytes(four_bytes);

    assert_eq!(num_images_train, num_labels_train);

    test_set_image_file.read_exact(&mut four_bytes).unwrap();
    let num_images_test = i32::from_be_bytes(four_bytes);
    test_set_label_file.read_exact(&mut four_bytes).unwrap();
    let num_labels_test = i32::from_be_bytes(four_bytes);

    assert_eq!(num_images_test, num_labels_test);

    // There are 8 more bytes in image data, ie. num_rows & num_cols, unsigned
    // byte
    training_set_image_file.read_exact(&mut four_bytes).unwrap();
    let num_rows_train_image = u32::from_be_bytes(four_bytes);
    training_set_image_file.read_exact(&mut four_bytes).unwrap();
    let num_cols_train_image = u32::from_be_bytes(four_bytes);

    test_set_image_file.read_exact(&mut four_bytes).unwrap();
    let num_rows_test_image = u32::from_be_bytes(four_bytes);
    test_set_image_file.read_exact(&mut four_bytes).unwrap();
    let num_cols_test_image = u32::from_be_bytes(four_bytes);

    // MNIST Dataset has 28*28 image data
    assert_eq!(28, num_rows_test_image);
    assert_eq!(28, num_cols_test_image);
    assert_eq!(28, num_rows_train_image);
    assert_eq!(28, num_cols_train_image);

    // DONE ALL VERIFYING... Now we can read the feature vectors (image data),
    // and labels (which alphabet it is, eg. A, B, C...)
    let mut one_byte = [0u8; 1];
    let mut image_bytes = Vec::new();
    image_bytes.resize( (num_rows_train_image * num_cols_train_image) as usize, 0u8);

    let mut labels = std::collections::HashSet::new();

    for _i in 0..num_images_train {
        training_set_label_file.read_exact(&mut one_byte).unwrap();
        let label = u8::from_be_bytes(one_byte);

        labels.insert(label);

        training_set_image_file.read_exact(&mut image_bytes).unwrap();
        
        train_dataset.push(
                Observation::new(label, &image_bytes)
            );
    }

    for _i in 0..num_images_test {
        test_set_label_file.read_exact(&mut one_byte).unwrap();
        let label = u8::from_be_bytes(one_byte);

        labels.insert(label);

        test_set_image_file.read_exact(&mut image_bytes).unwrap();

        test_dataset.push(
                Observation::new(label, &image_bytes)
            );
    }

    println!("Successfully read: {} train observations, and {} test observations !", train_dataset.len(), test_dataset.len() );

    println!("Total labels read: {}", labels.len());
    println!("Labels: {:?}", labels);

    let mut dataset = train_dataset;
    dataset.append(&mut test_dataset);

    let knn_model = KNN::new(&mut dataset);
}
