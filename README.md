# One-phase Batch Update on Sparse Merkle Tree for Rollups

This repository contains the implementation of the paper "One-phase Batch Update on Sparse Merkle Tree for Rollups". https://arxiv.org/abs/2310.13328


**Acknowledgement**: Thanks to the zkSync team from Matter Labs for the Sparse Merkle tree implementation of which this repository is based on. Refer to https://github.com/matter-labs/zksync for the original code base.

## Repository Structure
The file structure below only includes relevent directories
```
├── one-phase-batch-update-SMT
    ├── bin
        |── test_tree (Playground of various experiments)
        |── data (real-world data collected from zkSync Lite scanner)
        |── tree_cache (pre-loaded SMTs)
        |── src
            |── libs.rs (helper functions)
            |── main.rs (main experiment loop)
    ├── lib
        ├── crypto
            ├── src
                ├── merkle_tree
                    ├── parallel_smt.rs (Implementation of SMT algorithms)
```

## How to run
**Tests**
```
cd one-phase-batch-update-SMT
cargo test
``` 
**Experiments**
Before running the experiments, you will need to hard code the experiment you wish to run. Available experiments are:
1. `big_bench` (SMT.Update, refer to Fig.4.A1 in paper) 
2. `big_bench_random` (SMT.Update in random order, refer to Fig.4.A2 in paper)
3. `big_bench_insert_new`(SMT.Insert, refer to Fig.4.A3 in paper)
4. `big_bench_remove` (SMT.Remove)
5. `benchmark_real_blocks` (MacroBenchmark, refer to fig.6 in paper)
```
cd one-phase-batch-update-SMT/bin/test_tree
cargo run > results.txt
``` 
