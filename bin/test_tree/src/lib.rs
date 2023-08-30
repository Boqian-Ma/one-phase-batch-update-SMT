use rand::distributions::{Distribution, Uniform};
use {
    zksync_crypto::merkle_tree::{
        parallel_smt, parallel_smt::SMTLeafUpdate, parallel_smt::SMTLeafUpdates, RescueHasher,
    },
    zksync_crypto::Engine,
    zksync_crypto::Fr,
};

use csv::Reader;

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::{
    collections::HashMap,
    io::{self, BufRead},
};

use std::fs;
use zksync_crypto::merkle_tree::parallel_smt::SparseMerkleTreeSerializableCacheBN256;

// const TREE_DEPTH: u32 = 24;
const BASE: u32 = 2;

fn save_tree() {
    let depth = 24;

    let path: String = format!("bin/test_tree/tree_cache/{}_full_tree.txt", depth);

    let base_tree_1 = save_tree_cache(depth);
    let base_tree_2 = load_tree_cache(&path, depth);

    assert!(base_tree_1.root_hash() == base_tree_2.root_hash());

    assert!(
        base_tree_1.items.read().expect("Read lock").clone()
            == base_tree_2.items.read().expect("Read lock").clone()
    );

    assert!(
        base_tree_1.cache.read().expect("Read lock").clone()
            == base_tree_2.cache.read().expect("Read lock").clone()
    );

    for (key, val) in base_tree_1.index_node_map {
        if !base_tree_2.index_node_map.contains_key(&key)
            || base_tree_2.index_node_map[&key].index != val.index
        {
            panic!();
        }
    }
}
pub fn save_tree_cache(
    depth: u32,
) -> parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>> {
    let path: String = format!("bin/test_tree/tree_cache/{}_full_tree.txt", depth);

    if Path::new(&path).exists() {
        panic!("File already exists");
    }

    let mut tree =
        parallel_smt::SparseMerkleTree::<u64, Fr, RescueHasher<Engine>>::new(depth as usize);

    let balance: u64 = 100;

    // fill tree leafs
    for idx in 0..BASE.pow(depth) {
        tree.insert(idx, balance);
    }
    // cache root hash
    tree.root_hash();
    let cache_encode: Vec<u8> = tree.get_internals().encode_bincode();

    write_vector_to_file(&cache_encode, &path).unwrap();
    tree
}

pub fn load_tree_cache(
    file_path: &str,
    depth: u32,
) -> parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>> {
    let loaded_vec: Vec<u8> = load_vector_from_file(file_path).unwrap();
    let case_decode = SparseMerkleTreeSerializableCacheBN256::decode_bincode(&loaded_vec);
    let mut tree =
        parallel_smt::SparseMerkleTree::<u64, Fr, RescueHasher<Engine>>::new(depth as usize);
    tree.set_internals(case_decode);
    tree
}

fn write_vector_to_file(vector: &Vec<u8>, file_path: &str) -> std::io::Result<()> {
    fs::write(file_path, vector)
}

fn load_vector_from_file(file_path: &str) -> std::io::Result<Vec<u8>> {
    let data = fs::read(file_path)?;
    Ok(data)
}

pub fn generate_tree(
    depth: u32,
    num_leafs: u32,
) -> parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>> {
    // create two trees to compare
    let mut tree =
        parallel_smt::SparseMerkleTree::<u64, Fr, RescueHasher<Engine>>::new(depth as usize);
    let balance: u64 = 100;
    // for idx in 0..BASE.pow(depth) {

    for idx in 0..num_leafs {
        tree.insert(idx, balance);
    }

    // cache root hash
    tree.root_hash();
    tree
}

pub fn generate_tree_given_list(
    depth: u32,
    leafs_idx: Vec<u32>,
) -> parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>> {
    // create two trees to compare
    let mut tree =
        parallel_smt::SparseMerkleTree::<u64, Fr, RescueHasher<Engine>>::new(depth as usize);
    let balance: u64 = 100;

    for idx in leafs_idx.iter() {
        tree.insert(*idx, balance);
    }

    tree.root_hash();
    tree
}

pub fn get_smtleaf_updates(quantity: u32, balance: u64) -> SMTLeafUpdates<u64> {
    let mut updates: SMTLeafUpdates<u64> = vec![];
    for idx in 0..quantity {
        updates.push((
            idx as u64,
            SMTLeafUpdate::Insert {
                item_index: (idx as u64),
                item: balance as u64,
            },
        ));
    }
    updates
}

pub fn get_smtleaf_remove(quantity: u32) -> SMTLeafUpdates<u64> {
    let mut updates: SMTLeafUpdates<u64> = vec![];
    for idx in 0..quantity {
        updates.push((
            idx as u64,
            SMTLeafUpdate::Remove {
                item_index: (idx as u64),
            },
        ));
    }
    updates
}

pub fn get_smtleaf_updates_insert_new(quantity: u32, balance: u64) -> SMTLeafUpdates<u64> {
    let mut updates: SMTLeafUpdates<u64> = vec![];
    for idx in 0..quantity {
        updates.push((
            idx as u64,
            SMTLeafUpdate::InsertNew {
                item_index: (idx as u64),
                item: balance as u64,
            },
        ));
    }
    updates
}

// Random function
pub fn sample_random_accounts(total_leafs: u32, num_samples: u32) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    let die = Uniform::from(0..total_leafs); // set your range here
    let nums: Vec<u32> = (0..num_samples).map(|_| die.sample(&mut rng)).collect();
    nums
}

pub fn get_batch_updates_from_list(accounts: &Vec<u32>, balance: u64) -> SMTLeafUpdates<u64> {
    // let accounts = sample_random_accounts(total_leafs, samples);

    let mut updates: SMTLeafUpdates<u64> = vec![];

    for idx in accounts.iter() {
        updates.push((
            *idx as u64,
            SMTLeafUpdate::Insert {
                item_index: (*idx as u64),
                item: balance,
            },
        ));
    }

    updates
}

/// load json accounts into a map, id: account
/// return map
pub fn load_tree_from_csv(
    path: String,
) -> io::Result<parallel_smt::SparseMerkleTree<u64, Fr, RescueHasher<Engine>>> {
    let mut tree = generate_tree(24, 0);

    // Open the file
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read the values separated by commas and store them in a vector
    let mut values: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let tokens: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

        for token in tokens {
            values.push(token.to_string());
        }
    }

    let account_ids: Result<Vec<u32>, _> = values.iter().map(|s| s.parse::<u32>()).collect();
    let parsed_numbers = account_ids.unwrap();

    for id in parsed_numbers {
        tree.insert(id, 0);
    }

    tree.root_hash();
    Ok(tree)
}

pub fn load_block_data(path: String) -> io::Result<HashMap<u32, Vec<u32>>> {
    let mut data: HashMap<u32, Vec<u32>> = HashMap::new();

    let file = File::open(path)?;
    let mut csv_reader = Reader::from_reader(file);

    for result in csv_reader.records() {
        let record = result?;
        let block_id = &record[0];
        let from_account = &record[2]; // Index 2 corresponds to "From Account" column
        let to_account = &record[3]; // Index 3 corresponds to "To Account" column

        let entry = data
            .entry(block_id.parse::<u32>().unwrap())
            .or_insert(Vec::new());

        // push operations into the vector
        entry.push(from_account.parse::<u32>().unwrap());
        entry.push(to_account.parse::<u32>().unwrap());
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_block_data() {
        let path = String::from("/home/admin/zksync/core/bin/test_tree/data/block_data.csv");
        let data = load_block_data(path).unwrap();
        println!("{:?}", data.len());
    }

    #[test]
    fn test_load_tree() {
        let path = String::from("/home/admin/zksync/core/bin/test_tree/data/accounts.txt");
        let tree = load_tree_from_csv(path).unwrap();
        println!("{:?}", tree.index_node_map.len());
    }

    #[test]
    fn test_merle_tree_batch_remove_one() {
        let depth = 4;
        let num_leafs = 16;

        // Arrange
        let mut tree_1 = generate_tree(depth, num_leafs);
        let mut tree_2 = tree_1.clone();

        let idxs = vec![0];

        let mut updates: SMTLeafUpdates<u64> = vec![];
        for idx in idxs.iter() {
            updates.push((
                *idx as u64,
                SMTLeafUpdate::Remove {
                    item_index: (*idx as u64),
                },
            ));
        }

        // Act
        let tree_1_hash = tree_1.batch_insert(updates).unwrap();

        for idx in idxs.iter() {
            tree_2.remove(*idx);
        }
        let tree_2_hash = tree_2.root_hash();

        assert_eq!(tree_1_hash, tree_2_hash);
    }

    #[test]
    fn test_merle_tree_batch_remove_multiple_sqeuential() {
        let depth = 4;
        let num_leafs = 16;

        // Arrange
        let mut tree_1 = generate_tree(depth, num_leafs);
        let mut tree_2 = tree_1.clone();

        let idxs = vec![0, 1, 2];

        let mut updates: SMTLeafUpdates<u64> = vec![];
        for idx in idxs.iter() {
            updates.push((
                *idx as u64,
                SMTLeafUpdate::Remove {
                    item_index: (*idx as u64),
                },
            ));
        }

        // Act
        let tree_1_hash = tree_1.batch_insert(updates).unwrap();

        for idx in idxs.iter() {
            tree_2.remove(*idx);
        }
        let tree_2_hash = tree_2.root_hash();

        assert_eq!(tree_1_hash, tree_2_hash);
    }

    #[test]
    fn test_merle_tree_batch_remove_multiple_skip() {
        let depth = 4;
        let num_leafs = 16;

        // Arrange
        let mut tree_1 = generate_tree(depth, num_leafs);
        let mut tree_2 = tree_1.clone();

        let idxs = vec![0, 3, 5];

        let mut updates: SMTLeafUpdates<u64> = vec![];
        for idx in idxs.iter() {
            updates.push((
                *idx as u64,
                SMTLeafUpdate::Remove {
                    item_index: (*idx as u64),
                },
            ));
        }

        // Act
        let tree_1_hash = tree_1.batch_insert(updates).unwrap();

        for idx in idxs.iter() {
            tree_2.remove(*idx);
        }
        let tree_2_hash = tree_2.root_hash();

        assert_eq!(tree_1_hash, tree_2_hash);
    }

    #[test]
    fn test_merle_tree_batch_update_single() {
        let depth = 4;
        let num_leafs = 16;
        let item = 1;

        // Arrange
        let mut tree_1 = generate_tree(depth, num_leafs);
        let mut tree_2 = tree_1.clone();

        let idxs = vec![0];

        let mut updates: SMTLeafUpdates<u64> = vec![];
        for idx in idxs.iter() {
            updates.push((
                *idx as u64,
                SMTLeafUpdate::Insert {
                    item_index: (*idx as u64),
                    item: item,
                },
            ));
        }

        // Act
        let tree_1_hash = tree_1.batch_insert(updates).unwrap();

        for idx in idxs.iter() {
            tree_2.insert(*idx, item);
        }
        let tree_2_hash = tree_2.root_hash();

        assert_eq!(tree_1_hash, tree_2_hash);
    }

    #[test]
    fn test_merle_tree_batch_update_multiple() {
        let depth = 4;
        let num_leafs = 16;
        let item = 1;

        // Arrange
        let mut tree_1 = generate_tree(depth, num_leafs);
        let mut tree_2 = tree_1.clone();

        let idxs = vec![0, 1, 2, 4];

        let mut updates: SMTLeafUpdates<u64> = vec![];
        for idx in idxs.iter() {
            updates.push((
                *idx as u64,
                SMTLeafUpdate::Insert {
                    item_index: (*idx as u64),
                    item: item,
                },
            ));
        }

        // Act
        let tree_1_hash = tree_1.batch_insert(updates).unwrap();

        for idx in idxs.iter() {
            tree_2.insert(*idx, item);
        }
        let tree_2_hash = tree_2.root_hash();

        assert_eq!(tree_1_hash, tree_2_hash);
    }

    #[test]
    fn test_merle_tree_batch_insert_new_single() {
        let depth = 4;
        let item = 1;

        // Arrange
        let mut tree_1 = generate_tree(depth, 0);
        let mut tree_2 = tree_1.clone();

        let idxs = vec![0];

        let mut updates: SMTLeafUpdates<u64> = vec![];
        for idx in idxs.iter() {
            updates.push((
                *idx as u64,
                SMTLeafUpdate::InsertNew {
                    item_index: (*idx as u64),
                    item: item,
                },
            ));
        }

        // Act
        let tree_1_hash = tree_1.batch_insert(updates).unwrap();

        for idx in idxs.iter() {
            tree_2.insert(*idx, item);
        }
        let tree_2_hash = tree_2.root_hash();

        assert_eq!(tree_1_hash, tree_2_hash);
    }
    #[test]
    fn test_merle_tree_batch_insert_new_multiple() {
        let depth = 4;
        let item = 1;

        // Arrange
        let mut tree_1 = generate_tree(depth, 0);
        let mut tree_2 = tree_1.clone();

        let idxs = vec![0, 1, 2, 3, 8];

        let mut updates: SMTLeafUpdates<u64> = vec![];
        for idx in idxs.iter() {
            updates.push((
                *idx as u64,
                SMTLeafUpdate::InsertNew {
                    item_index: (*idx as u64),
                    item: item,
                },
            ));
        }

        // Act
        let tree_1_hash = tree_1.batch_insert(updates).unwrap();

        for idx in idxs.iter() {
            tree_2.insert(*idx, item);
        }
        let tree_2_hash = tree_2.root_hash();

        assert_eq!(tree_1_hash, tree_2_hash);
    }
}
