pub struct Observation {
    feature_vectors: Vec<u8>,
    label: u8,
    enum_label: u32 // an integral repr
}

impl Observation {
    pub fn new(label: u8, num_rows: i32, num_col: i32, image_data: &[u8]) -> Observation {
        
        todo!();
    }
}
