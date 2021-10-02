use crate::observation::Observation;
use rand::{thread_rng, seq::SliceRandom};

pub struct KNN<'a> {
    pub k: u32,
    dataset: &'a [Observation],  // complete dataset
    pub training_set: &'a [Observation],
    pub validation_set: &'a [Observation],
    pub test_set: &'a [Observation]
}

impl KNN<'_> {
    pub fn new<'a>(dataset: &'a mut [Observation]) -> KNN<'a> {

        const TRAIN_SET_RATIO: f32 = 0.75;   // use 75% of dataset for training
        const VALIDATION_SET_RATIO: f32 = 0.10;
        const TEST_SET_RATIO: f32 = 0.15;

        let num_elements_train = TRAIN_SET_RATIO * dataset.len() as f32;
        let num_elements_validation  = TEST_SET_RATIO * dataset.len() as f32;

        let mut rng = thread_rng();
        dataset.shuffle(&mut rng);
        let (training_set, rest_of_data)
            = dataset.split_at(num_elements_train as usize);

        let (validation_set, test_set)
            = rest_of_data.split_at(num_elements_validation as usize);

        KNN {
            k: 5,   // Won't be used, k needs to be calculated later
            dataset,
            training_set,
            validation_set,
            test_set
        }
    }
}

