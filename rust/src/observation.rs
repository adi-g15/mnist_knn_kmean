#[allow(dead_code)]

pub struct Observation {
    feature_vectors: Vec<u8>,
    label: u8,
}

impl Observation {
    #[allow(dead_code)]
    pub fn new(label: u8,/* num_rows: i32, num_col: i32,*/ image_data: &[u8])
        -> Observation {
        Observation {
            feature_vectors: Vec::from(image_data),
            label
        }
    }
}
