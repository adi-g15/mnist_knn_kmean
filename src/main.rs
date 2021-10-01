mod observation;

use observation::Observation;
use std::fs::File;
use std::io::Read;  // trait required for read functions on file

use std::io::Cursor;
use byteorder::{BigEndian,ReadBytesExt,LittleEndian};

use std::os::unix::fs::FileExt; // for converting bytes to number

fn check_if_magic_matches(bytes: &[u8;4], magic_number: u32) -> Result<(),String> {
    let mut reader = Cursor::new(bytes);

    let read_number = reader.read_u32::<BigEndian>().unwrap();
    if magic_number == read_number {
        Ok(())
    } else {
        Err(format!("Magic number doesn't match: Expected={}, Read={}", magic_number, read_number))
    }
}

fn main() {
    let train_dataset: Vec<Observation> = Vec::new();
    let test_dataset: Vec<Observation> = Vec::new();

    let mut training_set_label_file = File::open("dataset/train-labels-idx1-ubyte").unwrap();
    let mut training_set_image_file = File::open("dataset/train-images-idx3-ubyte").unwrap();
    let mut test_set_label_file = File::open("dataset/t10k-labels-idx1-ubyte").unwrap();
    let mut test_set_image_file = File::open("dataset/t10k-images-idx3-ubyte").unwrap();

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
}
