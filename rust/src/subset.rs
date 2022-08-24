    // Idea from the "Competetive programmer handbook", we can use a binary
    // number to denote a subset, with 1 for indices of elements included in
    // this 'subset' and 0 for indices not in this subset

use observation::Observation;
use byte_set::ByteSet;

struct Subset<'a> {
    original_vec: &'a [Observation],
    _subset_repr: ByteSet
}

impl Subset<'_> {
    pub fn new<'a>(original_vec: &[Observation]) -> Subset {
        Subset {
            original_vec,
            _subset_repr: ByteSet::new()
        }
    }
}
