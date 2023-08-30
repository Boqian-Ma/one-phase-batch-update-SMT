use criterion::criterion_main;

use insert::insert_benches;

mod insert;

// create trees and insert acounts
// dont need fake hashers anymore

criterion_main!(insert_benches);
