use std::{collections::HashMap, time::Instant};

use std::hint::black_box;

use test_tree::{
    generate_tree, generate_tree_given_list, get_batch_updates_from_list, get_smtleaf_remove,
    get_smtleaf_updates, get_smtleaf_updates_insert_new, load_block_data, load_tree_from_csv,
    sample_random_accounts,
};

use {
    zksync_crypto::merkle_tree::{parallel_smt, RescueHasher},
    zksync_crypto::Engine,
    zksync_crypto::Fr,
};

fn benchmark_normal(
    num_leaf_updates: u32,
    tree: parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>>,
) -> (f32, Vec<u32>) {
    let mut results: Vec<u32> = vec![];

    // take average of ten runs
    for i in 0..10 {
        let mut curr_tree = tree.clone();

        println!("Starting normal insert: {}", i);
        let start = Instant::now();
        // insert
        for idx in 0..num_leaf_updates {
            // curr_tree.insert(black_box(idx as u32), black_box(10));
            curr_tree.insert(idx as u32, 10);
        }
        let insert_time = start.elapsed().as_millis();
        println!("Insert time: {}", insert_time);

        curr_tree.root_hash();

        let end = Instant::now();
        let time = end.duration_since(start).as_micros();
        println!("Normal insert: {}: {}", i, time);
        results.push(time as u32);
    }

    println!("Normal insert: {:?}, {}", results, results.len());

    let sum: u32 = results.iter().sum(); // Calculate the sum of all elements
    let average: f32 = sum as f32 / results.len() as f32; // Calculate the average
    println!("Average: {} microseconds", average);

    (average, results)
}

fn benchmark_normal_remove(
    num_leaf_updates: u32,
    tree: parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>>,
) -> (f32, Vec<u32>) {
    let mut results: Vec<u32> = vec![];

    // take average of ten runs
    for i in 0..10 {
        let mut curr_tree = tree.clone();

        println!("Starting normal insert: {}", i);
        let start = Instant::now();
        // insert
        for idx in 0..num_leaf_updates {
            // curr_tree.insert(black_box(idx as u32), black_box(10));
            curr_tree.remove(idx as u32);
        }
        let insert_time = start.elapsed().as_millis();
        println!("Insert time: {}", insert_time);

        curr_tree.root_hash();

        let end = Instant::now();
        let time = end.duration_since(start).as_micros();
        println!("Normal insert: {}: {}", i, time);
        results.push(time as u32);
    }

    println!("Normal insert: {:?}, {}", results, results.len());

    let sum: u32 = results.iter().sum(); // Calculate the sum of all elements
    let average: f32 = sum as f32 / results.len() as f32; // Calculate the average
    println!("Average: {} microseconds", average);

    (average, results)
}

fn benchmark_batch_remove(
    num_leaf_updates: u32,
    tree: parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>>,
) -> (f32, Vec<u32>) {
    // single thread, take 10 measurements microseconds, average
    // let start_load = Instant::now();
    let mut results: Vec<u32> = vec![];

    for i in 0..10 {
        let mut curr_tree = tree.clone();
        let updates = get_smtleaf_remove(num_leaf_updates);
        println!("Starting batch remove: {}", i);
        let start = Instant::now();
        let _ = curr_tree.batch_insert(black_box(updates.clone()));
        let end = Instant::now();
        let time = end.duration_since(start).as_micros();
        println!("Batch remove {}: {}", i, time);
        results.push(time as u32);
    }

    println!("Batch insert: {:?}, {}", results, results.len());

    let sum: u32 = results.iter().sum(); // Calculate the sum of all elements
    let average: f32 = sum as f32 / results.len() as f32; // Calculate the average
    println!("Average: {} microseconds", average);

    (average, results)
}

fn benchmark_batch_insert(
    num_leaf_updates: u32,
    tree: parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>>,
) -> (f32, Vec<u32>) {
    // single thread, take 10 measurements microseconds, average
    // let start_load = Instant::now();
    let mut results: Vec<u32> = vec![];

    for i in 0..10 {
        let mut curr_tree = tree.clone();
        let updates = get_smtleaf_updates(num_leaf_updates, 10);
        println!("Starting batch insert: {}", i);
        let start = Instant::now();
        let _ = curr_tree.batch_insert(black_box(updates.clone()));
        let end = Instant::now();
        let time = end.duration_since(start).as_micros();
        println!("Batch insert {}: {}", i, time);
        results.push(time as u32);
    }

    println!("Batch insert: {:?}, {}", results, results.len());

    let sum: u32 = results.iter().sum(); // Calculate the sum of all elements
    let average: f32 = sum as f32 / results.len() as f32; // Calculate the average
    println!("Average: {} microseconds", average);

    (average, results)
}

fn benchmark_batch_insert_new(
    num_leaf_updates: u32,
    tree: parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>>,
) -> (f32, Vec<u32>) {
    // single thread, take 10 measurements microseconds, average
    // let start_load = Instant::now();
    let mut results: Vec<u32> = vec![];

    for i in 0..10 {
        let mut curr_tree = tree.clone();

        let updates = get_smtleaf_updates_insert_new(num_leaf_updates, 10);
        println!("Starting batch insert: {}", i);
        let start = Instant::now();
        let _ = curr_tree.batch_insert(black_box(updates.clone()));
        let end = Instant::now();
        let time = end.duration_since(start).as_micros();
        println!("Batch insert {}: {}", i, time);
        results.push(time as u32);
    }

    println!("Batch insert: {:?}, {}", results, results.len());

    let sum: u32 = results.iter().sum(); // Calculate the sum of all elements
    let average: f32 = sum as f32 / results.len() as f32; // Calculate the average
    println!("Average: {} microseconds", average);

    (average, results)
}

fn benchmark_normal_random(total_leafs: u32, num_leaf: u32, depth: u32) -> (f32, Vec<u32>) {
    let mut results: Vec<u32> = vec![];

    // take average of ten runs
    for i in 0..10 {
        let accounts = sample_random_accounts(total_leafs, num_leaf);
        let base_tree = generate_tree_given_list(depth, accounts.clone());

        let mut curr_tree = base_tree.clone();
        // let accounts = sample_random_accounts(total_leafs, num_leaf);

        println!("Starting normal insert: {}", i);
        let start = Instant::now();
        // insert
        for idx in accounts.iter() {
            // curr_tree.insert(black_box(idx as u32), black_box(10));
            curr_tree.insert(*idx as u32, 10);
        }
        curr_tree.root_hash();

        let end = Instant::now();
        let time = end.duration_since(start).as_micros();
        println!("Normal insert: {}: {}", i, time);
        results.push(time as u32);
    }

    println!("Normal insert: {:?}, {}", results, results.len());

    let sum: u32 = results.iter().sum(); // Calculate the sum of all elements
    let average: f32 = sum as f32 / results.len() as f32; // Calculate the average
    println!("Average: {} microseconds", average);

    (average, results)
}

fn benchmark_batch_random(
    total_leafs: u32,
    num_leaf: u32,
    depth: u32, // tree: parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>>,
) -> (f32, Vec<u32>) {
    let mut results: Vec<u32> = vec![];

    for i in 0..10 {
        let accounts = sample_random_accounts(total_leafs, num_leaf);
        let base_tree = generate_tree_given_list(depth, accounts.clone());

        let updates = get_batch_updates_from_list(&accounts, 10);
        assert!(updates.len() == accounts.len());

        println!("Starting batch insert: {}", i);
        let mut curr_tree = base_tree.clone();
        let start = Instant::now();
        let _ = curr_tree.batch_insert(black_box(updates.clone()));
        let end = Instant::now();
        let time = end.duration_since(start).as_micros();

        println!("Batch insert {}: {}", i, time);
        results.push(time as u32);
    }

    println!("Batch insert: {:?}, {}", results, results.len());
    let sum: u32 = results.iter().sum(); // Calculate the sum of all elements
    let average: f32 = sum as f32 / results.len() as f32; // Calculate the average
    println!("Average: {} microseconds", average);
    (average, results)
}

fn big_bench_random(depth: u32) {
    let num_leaf_inserted = vec![
        1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        1700, 1800, 1900, 2000,
    ];

    let mut res: HashMap<String, f32> = HashMap::new();
    let mut data_res: HashMap<String, Vec<u32>> = HashMap::new();

    // batch insert data
    let mut res_batch: HashMap<String, f32> = HashMap::new();
    let mut data_res_batch: HashMap<String, Vec<u32>> = HashMap::new();

    let total_leafs = 2u32.pow(depth);

    for num_leaf in num_leaf_inserted.iter() {
        let (avg, data) = benchmark_normal_random(total_leafs, *num_leaf, depth);
        let (avg_batch, data_batch) = benchmark_batch_random(total_leafs, *num_leaf, depth);

        res.insert(num_leaf.to_string(), avg);
        data_res.insert(num_leaf.to_string(), data);

        res_batch.insert(num_leaf.to_string(), avg_batch);
        data_res_batch.insert(num_leaf.to_string(), data_batch);
    }

    let mut vector: Vec<(String, f32)> = res.into_iter().collect();
    vector.sort_by_key(|(key, _)| key.clone());

    for (key, value) in &vector {
        println!(
            "# Inserts {}:, {}, {:?}",
            key,
            value,
            res_batch.get(key).unwrap()
        );
    }

    for (key, _) in &vector {
        println!("# Inserts {}", key);
        println!("Data: {:?}", data_res.get(key).unwrap());
        println!("Batch Data: {:?}", data_res_batch.get(key).unwrap());
    }

    // get random number list
}

fn big_bench(depth: u32) {
    let num_leaf_inserted = vec![
        1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
    ];

    // normal insert data
    let mut res: HashMap<String, f32> = HashMap::new();
    let mut data_res: HashMap<String, Vec<u32>> = HashMap::new();

    // batch insert data
    let mut res_batch: HashMap<String, f32> = HashMap::new();
    let mut data_res_batch: HashMap<String, Vec<u32>> = HashMap::new();
    for num_leaf in num_leaf_inserted.iter() {
        // start new tree

        let tree = generate_tree(depth, *num_leaf);

        println!("***Starting benchmark: {}***", num_leaf);
        let (avg, data) = benchmark_normal(*num_leaf, tree.clone());
        let (avg_batch, data_batch) = benchmark_batch_insert(*num_leaf, tree.clone());

        res.insert(num_leaf.to_string(), avg);
        data_res.insert(num_leaf.to_string(), data);

        res_batch.insert(num_leaf.to_string(), avg_batch);
        data_res_batch.insert(num_leaf.to_string(), data_batch);
    }

    let mut vector: Vec<(String, f32)> = res.into_iter().collect();
    vector.sort_by_key(|(key, _)| key.clone());
    for (key, value) in &vector {
        println!(
            "# Inserts {}:, {}, {:?}",
            key,
            value,
            res_batch.get(key).unwrap()
        );
    }

    for (key, _) in &vector {
        println!("# Inserts {}", key);
        println!("Data: {:?}", data_res.get(key).unwrap());
        println!("Batch Data: {:?}", data_res_batch.get(key).unwrap());
    }
}

/// This function is used to test the performance of insert new
fn big_bench_insert_new(depth: u32) {
    // Prepare empty tree
    let base_tree = generate_tree(depth, 0);

    // check insert new speed
    let num_leaf_inserted = vec![
        1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000,
    ];

    // normal insert data
    let mut res: HashMap<String, f32> = HashMap::new();
    let mut data_res: HashMap<String, Vec<u32>> = HashMap::new();

    let mut res_batch: HashMap<String, f32> = HashMap::new();
    let mut data_res_batch: HashMap<String, Vec<u32>> = HashMap::new();

    for num_leaf in num_leaf_inserted.iter() {
        println!("***Starting benchmark: {}***", num_leaf);
        let (avg, data) = benchmark_normal(*num_leaf, base_tree.clone());

        let (avg_batch, data_batch) = benchmark_batch_insert_new(*num_leaf, base_tree.clone());

        res.insert(num_leaf.to_string(), avg);
        data_res.insert(num_leaf.to_string(), data);

        res_batch.insert(num_leaf.to_string(), avg_batch);
        data_res_batch.insert(num_leaf.to_string(), data_batch);
    }

    let mut vector: Vec<(String, f32)> = res.into_iter().collect();
    vector.sort_by_key(|(key, _)| key.clone());

    for (key, value) in &vector {
        println!(
            "# Inserts {}:, {}, {:?}",
            key,
            value,
            res_batch.get(key).unwrap()
        );
    }

    for (key, _) in &vector {
        println!("# Inserts {}", key);
        println!("Data: {:?}", data_res.get(key).unwrap());
        println!("Batch Data: {:?}", data_res_batch.get(key).unwrap());
    }

    // insert new
}

fn big_bench_remove(depth: u32) {
    let num_leaf_inserted = vec![
        1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000,
    ];

    // normal insert data
    let mut res: HashMap<String, f32> = HashMap::new();
    let mut data_res: HashMap<String, Vec<u32>> = HashMap::new();

    let mut res_batch: HashMap<String, f32> = HashMap::new();
    let mut data_res_batch: HashMap<String, Vec<u32>> = HashMap::new();

    for num_leaf in num_leaf_inserted.iter() {
        let base_tree = generate_tree(depth, *num_leaf);

        println!("***Starting benchmark: {}***", num_leaf);
        let (avg, data) = benchmark_normal_remove(*num_leaf, base_tree.clone());

        let (avg_batch, data_batch) = benchmark_batch_remove(*num_leaf, base_tree.clone());

        res.insert(num_leaf.to_string(), avg);
        data_res.insert(num_leaf.to_string(), data);

        res_batch.insert(num_leaf.to_string(), avg_batch);
        data_res_batch.insert(num_leaf.to_string(), data_batch);
    }

    let mut vector: Vec<(String, f32)> = res.into_iter().collect();
    vector.sort_by_key(|(key, _)| key.clone());

    for (key, value) in &vector {
        println!(
            "# Inserts {}:, {}, {:?}",
            key,
            value,
            res_batch.get(key).unwrap()
        );
    }

    for (key, _) in &vector {
        println!("# Inserts {}", key);
        println!("Data: {:?}", data_res.get(key).unwrap());
        println!("Batch Data: {:?}", data_res_batch.get(key).unwrap());
    }
}

fn benchmark_insert_txs_list(
    accounts: &Vec<u32>,
    tree: parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>>,
) -> (f32, Vec<u32>) {
    let mut results: Vec<u32> = vec![];

    for i in 0..10 {
        let mut curr_tree = tree.clone();

        println!("Starting normal insert: {}", i);
        let start = Instant::now();
        // update given accounts
        for idx in accounts.iter() {
            // curr_tree.insert(black_box(idx as u32), black_box(10));
            curr_tree.insert(black_box(*idx) as u32, 100);
        }
        curr_tree.root_hash();

        let end = Instant::now();
        let time = end.duration_since(start).as_micros();
        println!("Normal insert: {}: {}", i, time);
        results.push(time as u32);
    }

    println!("Normal insert: {:?}, {}", results, results.len());
    let sum: u32 = results.iter().sum(); // Calculate the sum of all elements
    let average: f32 = sum as f32 / results.len() as f32; // Calculate the average
    println!("Average: {} microseconds", average);

    (average, results)
}

fn benchmark_batch_insert_txs_list(
    accounts: &Vec<u32>,
    tree: parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>>,
) -> (f32, Vec<u32>) {
    let mut results: Vec<u32> = vec![];

    for i in 0..10 {
        let mut curr_tree = tree.clone();
        println!("Starting batch insert: {}", i);

        let start = Instant::now();
        let updates = get_batch_updates_from_list(accounts, 100);
        let _ = curr_tree.batch_insert(black_box(updates.clone()));
        let end = Instant::now();

        let time = end.duration_since(start).as_micros();

        println!("Batch remove: {}, time: {}", i, time);
        results.push(time as u32);
    }

    println!("Normal insert: {:?}, {}", results, results.len());
    let sum: u32 = results.iter().sum(); // Calculate the sum of all elements
    let average: f32 = sum as f32 / results.len() as f32; // Calculate the average
    println!("Average: {} microseconds", average);

    (average, results)
}

fn benchmark_real_blocks() {
    let accounts_path = String::from("bin/test_tree/data/accounts.txt");
    let base_tree = load_tree_from_csv(accounts_path).unwrap();

    let block_data_path = String::from("bin/test_tree/data/block_data.csv");
    let block_data = load_block_data(block_data_path).unwrap();

    let mut res: HashMap<u32, f32> = HashMap::new();
    let mut data_res: HashMap<u32, Vec<u32>> = HashMap::new();

    let mut res_batch: HashMap<u32, f32> = HashMap::new();
    let mut data_res_batch: HashMap<u32, Vec<u32>> = HashMap::new();

    for (block_id, transactions) in block_data.iter() {
        println!("Starting Block: {}", block_id);
        let (avg, data) = benchmark_insert_txs_list(transactions, base_tree.clone());
        let (avg_batch, data_batch) =
            benchmark_batch_insert_txs_list(transactions, base_tree.clone());

        res.insert(*block_id, avg);
        data_res.insert(*block_id, data);

        res_batch.insert(*block_id, avg_batch);
        data_res_batch.insert(*block_id, data_batch);
    }

    let mut vector: Vec<(u32, f32)> = res.into_iter().collect();
    vector.sort_by_key(|(key, _)| key.clone());

    for (key, value) in &vector {
        println!(
            "# Block {}:, {}, {:?}",
            key,
            value,
            res_batch.get(key).unwrap()
        );
    }

    for (key, _) in &vector {
        println!("# Inserts {}", key);
        println!("Data: {:?}", data_res.get(key).unwrap());
        println!("Batch Data: {:?}", data_res_batch.get(key).unwrap());
    }
}

fn main() {
    let depth = 24;
    big_bench(depth);
    big_bench_random(depth);
    big_bench_insert_new(depth);
    big_bench_remove(depth);
    benchmark_real_blocks();
}
