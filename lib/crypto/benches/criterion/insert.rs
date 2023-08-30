//! Benchmarks for the `ZkSyncState` operations execution time.

use criterion::{black_box, criterion_group, BatchSize, Bencher, Criterion};

use {
    zksync_crypto::merkle_tree::{
        parallel_smt, parallel_smt::SMTLeafUpdate, parallel_smt::SMTLeafUpdates, RescueHasher,
    },
    // zksync_crypto::rand::{Rng, SeedableRng, XorShiftRng},
    zksync_crypto::Engine,
    zksync_crypto::Fr,
};

// use serde::{Deserialize, Serialize};

const TREE_DEPTH: u32 = 4;

const BASE: u32 = 2;

/// Generate tree and state
fn generate_tree() -> parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>> {
    // create two trees to compare
    let mut tree =
        parallel_smt::SparseMerkleTree::<u64, Fr, RescueHasher<Engine>>::new(TREE_DEPTH as usize);
    let balance: u64 = 100;
    for idx in 0..BASE.pow(TREE_DEPTH) {
        tree.insert(idx, balance);
    }
    // cache root hash
    tree.root_hash();
    tree
}

fn apply_normal_insert_bench(b: &mut Bencher<'_>) {
    let mut tree = generate_tree();
    // let setup = || tree.clone();
    b.iter(|| {
        // Inner closure, the actual test
        for idx in 0..BASE.pow(TREE_DEPTH) {
            tree.insert(idx, black_box(10));
        }
        // cache root hash
        tree.root_hash();
    });
}

fn get_SMTLeafUpdates() -> SMTLeafUpdates<u64> {
    let mut updates: SMTLeafUpdates<u64> = vec![];
    for idx in 0..BASE.pow(TREE_DEPTH) {
        updates.push((
            idx as u64,
            SMTLeafUpdate::Insert {
                item_index: (idx as u64),
                item: 10 as u64,
            },
        ));
    }

    updates
}

fn apply_batch_insert_bench(b: &mut Bencher<'_>) {
    let mut tree = generate_tree();

    // let mut updates: SMTLeafUpdates<u64> = vec![];
    let updates: SMTLeafUpdates<u64> = vec![
        (
            0,
            SMTLeafUpdate::Insert {
                item_index: (0),
                item: 10 as u64,
            },
        ),
        (
            1,
            SMTLeafUpdate::Insert {
                item_index: (1),
                item: 10 as u64,
            },
        ),
    ];

    // let setup = || tree.clone();
    // let c = updates.clone();
    b.iter_batched_ref(
        || -> SMTLeafUpdates<u64> { get_SMTLeafUpdates() },
        |v| tree.batch_insert(v.to_vec()),
        BatchSize::SmallInput,
    )
}

pub fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("SMT Insert Bench");

    group.bench_function(
        "parallel_smt::apply_insert bench",
        apply_normal_insert_bench,
    );

    group.bench_function(
        "parallel_smt::apply_batch_insert bench",
        apply_batch_insert_bench,
    );

    group.finish();
}

criterion_group!(insert_benches, bench_insert);
