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
    fn test_train_split(dataset: & mut [Observation]) ->
        (&[Observation], &[Observation], &[Observation]) {
        const TRAIN_SET_RATIO: f32 = 0.75;   // use 75% of dataset for training
        const VALIDATION_SET_RATIO: f32 = 0.10;
        const TEST_SET_RATIO: f32 = 0.15;

        let num_elements_train = TRAIN_SET_RATIO * dataset.len() as f32;
        let num_elements_test  = TEST_SET_RATIO * dataset.len() as f32;

        // TODO: Later completely shuffle the vector instead of partial_shuffle

        let mut rng = thread_rng();
        let (training_set, rest_of_data)
            = dataset.partial_shuffle(&mut rng, num_elements_train as usize);

        let (validation_set, test_set)
            = rest_of_data.partial_shuffle(&mut rng, num_elements_test as usize);

        (training_set, validation_set, test_set)
    }

    pub fn new<'a>(dataset: &'a mut [Observation]) -> KNN<'a> {
        let (training_set, validation_set, test_set);
            {
                let (a,b,c) = KNN::test_train_split(dataset);
                training_set = a;
                validation_set = b;
                test_set = c;
            };

        let data = dataset;

        KNN {
            k: 5,   // Won't be used, k needs to be calculated later
            dataset,
            training_set,
            validation_set,
            test_set
        }
    }
}

