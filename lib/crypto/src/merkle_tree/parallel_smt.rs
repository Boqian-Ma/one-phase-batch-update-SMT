/// Sparse Merkle tree with batch updates
use super::{hasher::Hasher, TreeMemoryUsage};
use crate::{
    ff::{PrimeField, PrimeFieldRepr},
    primitives::GetBits,
    Fr,
};

use bincode::Decode;
use fnv::FnvHashMap;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    sync::{RwLock, RwLockReadGuard},
};

use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Copy)]
pub enum SMTLeafUpdate<T> {
    Insert { item_index: ItemIndex, item: T },
    InsertNew { item_index: ItemIndex, item: T },
    Remove { item_index: ItemIndex },
}

pub type SMTLeafUpdates<T> = Vec<(ItemIndex, SMTLeafUpdate<T>)>;

impl<T, Hash, H> SparseMerkleTree<T, Hash, H>
where
    T: GetBits + Default + Sync + Send + Copy,
    Hash: Clone + Debug + Sync + Send + Copy,
    H: Hasher<Hash> + Sync + Send,
{
    pub fn wipe_cache_count(&mut self) {
        self.cache_hit_reads = 0;
        self.cache_hit_writes = 0;
    }

    pub fn batch_insert(&mut self, leaf_updates: SMTLeafUpdates<T>) -> Result<Hash, String> {
        self.batch_insert_parallel(leaf_updates)
    }

    fn batch_insert_parallel(&mut self, leaf_updates: SMTLeafUpdates<T>) -> Result<Hash, String> {
        let parent_level_map: RwLock<FnvHashMap<NodeIndex, bool>> =
            RwLock::new(FnvHashMap::default());
        let tree_depth = self.tree_depth;

        // Lines 5-9 in Algorithm 1 in paper
        leaf_updates.par_iter().for_each(|(item_index, update)| {
            assert!(*item_index < self.capacity());
            let leaf_index: NodeIndex = NodeIndex((1 << tree_depth) + item_index);
            match update {
                // insert
                SMTLeafUpdate::Insert { item_index, item } => {
                    // self.items.insert(item_index, item);
                    self.items
                        .write()
                        .expect("write lock")
                        .insert(*item_index, *item);

                    let direct_parent_index = NodeIndex((leaf_index.0) >> 1);

                    // insert parent to queue, could insert multiple times, but it's ok because we need to check anyway.
                    parent_level_map
                        .write()
                        .expect("write lock")
                        .insert(direct_parent_index, true);
                }
                // take care of insert new later in a non parallel way
                SMTLeafUpdate::Remove { item_index } => {
                    // remove from cache, replace with default hash
                    let default_item = T::default();
                    self.items
                        .write()
                        .expect("write lock")
                        .insert(*item_index, default_item);

                    let direct_parent_index = NodeIndex((leaf_index.0) >> 1);

                    // insert parent to queue, could insert multiple times, but it's ok because we need to check anyway.
                    parent_level_map
                        .write()
                        .expect("write lock")
                        .insert(direct_parent_index, true);
                }
                SMTLeafUpdate::InsertNew { item_index, item } => (),
            }
        });

        leaf_updates.iter().for_each(|(item_index, update)| {
            assert!(*item_index < self.capacity());
            let leaf_index: NodeIndex = NodeIndex((1 << tree_depth) + item_index);

            match update {
                // take care of insert new later in a non parallel way
                SMTLeafUpdate::InsertNew { item_index, item } => {
                    // insert() already invalidate all parents....., this will effect next loop.....
                    self.insert(*item_index as u32, *item);
                    // add direct parent to queue
                    // reconsider this because parents are invalidated in insert()
                    let direct_parent_index = NodeIndex((leaf_index.0) >> 1);
                    // remove parent cache and add parent to queue
                    self.cache
                        .write()
                        .expect("write lock")
                        .remove(&direct_parent_index);

                    // insert parent to queue, could insert multiple times
                    parent_level_map
                        .write()
                        .expect("write lock")
                        .insert(direct_parent_index, true);
                }
                SMTLeafUpdate::Insert {
                    item_index: _,
                    item: _,
                } => (),
                SMTLeafUpdate::Remove { item_index: _ } => (),
            }
        });

        // PARALLEL
        let leaf_updates_results: Vec<(NodeIndex, Hash)> = leaf_updates
            .par_iter()
            .filter_map(|(item_index, update)| {
                let leaf_index: NodeIndex = NodeIndex((1 << tree_depth) + item_index);
                match update {
                    // insert
                    SMTLeafUpdate::Insert { item_index, item }
                    | SMTLeafUpdate::InsertNew { item_index, item } => {
                        // hash leafs, this should happen in parallel
                        let item_bits = item.get_bits_le();
                        let item_hash = self.hasher.hash_bits(item_bits);
                        Some((leaf_index, item_hash))
                    }
                    // remove
                    SMTLeafUpdate::Remove { item_index } => {
                        // return default hash
                        let default_hash_for_layer = self.prehashed[self.tree_depth].clone();
                        Some((leaf_index, default_hash_for_layer))
                    }
                }
            })
            .collect();

        // NO PARALLEL
        for (node_idx, hash) in leaf_updates_results.iter() {
            self.cache
                .write()
                .expect("write lock")
                .insert(*node_idx, *hash);
        }

        let mut queue: VecDeque<NodeIndex> = parent_level_map
            .write()
            .expect("write lock")
            .keys()
            .cloned()
            .collect();

        // Lines 10-18 of Algorithm 1 in paper
        while !queue.is_empty() {
            // used to track parent queue
            let mut parent_level_map: HashMap<NodeIndex, bool> = HashMap::new();

            // current level
            let level_nodes: Vec<_> = queue.drain(..).collect();

            let level_nodes_hash_results: Vec<(NodeIndex, Hash)> = level_nodes
                .par_iter()
                .filter_map(|node_index| {
                    if let Some(front_node) = self.index_node_map.get(&node_index) {
                        // get children hash, should be in cache, else need to calculate

                        let left_hashes = self.get_child_hash(front_node, NodeDirection::Left);
                        let right_hashes = self.get_child_hash(front_node, NodeDirection::Right);

                        let (lhs_hash, _) = left_hashes;
                        let (rhs_hash, _) = right_hashes;

                        // // hash of current node
                        let hash = self.calculate_hash(front_node.depth, &lhs_hash, &rhs_hash);

                        Some((*node_index, hash))
                    } else {
                        // In this case, due to sparseness, we encounter a node that we didn't keep in record
                        // these nodes have only one child leaf, and is not a common parent to two leafs.
                        let left_child_idx = NodeIndex(node_index.0 << 1);
                        let right_child_idx = NodeIndex((node_index.0 << 1) + 1);
                        let curr_depth = Self::depth(*node_index);
                        let child_pre_hash = self.prehashed[curr_depth + 1].clone();

                        let cache_read_lock = self.cache.read().expect("Read lock");
                        let left_hashes = cache_read_lock.get(&left_child_idx);
                        let right_hashes = cache_read_lock.get(&right_child_idx);

                        // logic to find child hashes
                        match (left_hashes, right_hashes) {
                            (Some(lhs_hash), Some(rhs_hash)) => {
                                let hash = self.calculate_hash(curr_depth, lhs_hash, rhs_hash);
                                Some((*node_index, hash))
                            }
                            (Some(lhs_hash), None) => {
                                let hash =
                                    self.calculate_hash(curr_depth, lhs_hash, &child_pre_hash);
                                Some((*node_index, hash))
                            }
                            (None, Some(rhs_hash)) => {
                                let hash =
                                    self.calculate_hash(curr_depth, &child_pre_hash, rhs_hash);
                                Some((*node_index, hash))
                            }
                            _ => None,
                        }
                    }
                })
                .collect();

            // update cache
            for (node_idx, hash) in level_nodes_hash_results.iter() {
                let mut cache = self.cache.write().expect("write lock");
                cache.insert(*node_idx, *hash);
                let direct_parent_index = NodeIndex((node_idx.0) >> 1);
                // node index start from 1, dont need to put zero there
                if direct_parent_index.0 > 0 {
                    // remove parent cache, add parent to hashqueue
                    cache.remove(&direct_parent_index);
                    parent_level_map.insert(direct_parent_index, true);
                }
            }

            let mut next_queue = parent_level_map
                .keys()
                .cloned()
                .collect::<VecDeque<NodeIndex>>();

            std::mem::swap(&mut queue, &mut next_queue);
        }

        let root_hash = match self.cache.read().expect("Read lock").get(&NodeIndex(1)) {
            Some(cache) => {
                // self.cache_hit_reads += 1;
                cache.clone()
            }
            None => {
                return Err("Did not hit root hash cache".to_string());
            }
        };

        Ok(root_hash)
    }
}

/// Nodes are indexed starting with index(root) = 0
/// To store the index, at least 2 * TREE_HEIGHT bits is required.
/// Wrapper-structure is used to avoid mixing up with `ItemIndex` on the type level.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    bincode::Encode,
    bincode::Decode,
)]
pub struct NodeIndex(pub u64);

/// Lead index: 0 <= i < N.
type ItemIndex = u64;

/// Tree of depth 0: 1 item (which is root), level 0 only
/// Tree of depth 1: 2 items, levels 0 and 1
/// Tree of depth N: 2 ^ N items, 0 <= level < depth
type Depth = usize;

/// Index of the node in the vector; slightly inefficient, won't be needed when rust gets non-lexical lifetimes.
type NodeRef = usize;

/// Sparse Merkle tree with the support of the parallel hashes calculation.
///
/// Sparse Merkle tree is basically a [Merkle tree] which is allowed to have
/// gaps between elements.
///
/// The essential operation of this structure is obtaining a root hash of the structure,
/// which represents the state of all of the tree elements.
///
/// The sparseness of the tree is implementing through a "default leaf" - an item which
/// hash will be used for the missing indices instead of the actual element hash.
///
/// Since this means that basically the tree is "full" all the time (all the empty indices
/// are taken by the "default" element), the tree has fixed capacity and cannot be extended
/// above that. The root hash is calculated for the full tree every time.
///
/// [Merkle tree]: https://en.wikipedia.org/wiki/Merkle_tree
#[derive(Debug)]
pub struct SparseMerkleTree<T, Hash, H>
where
    T: GetBits,
    Hash: Clone + Debug,
    H: Hasher<Hash>,
{
    /// List of the stored items.
    // pub items: FnvHashMap<ItemIndex, T>,
    pub items: RwLock<FnvHashMap<ItemIndex, T>>,

    // items_temp: FnvHashMap<ItemIndex, T>,

    // self.items.write().expect("write lock").insert(leaf_index, item_hash);
    /// Generic hasher for the hash calculation.
    pub hasher: H,
    /// Fixed depth of the tree, determining the overall tree capacity.
    tree_depth: Depth,
    // Local index of the root node.
    root: NodeRef,
    // root index
    root_index: NodeIndex,
    // List of the intermediate nodes.
    nodes: Vec<Node>,
    // nodes: RwLock<Vec<Node>>,

    // Index to node map
    // pub index_node_map: RwLock<FnvHashMap<NodeIndex, Node>>,
    pub index_node_map: FnvHashMap<NodeIndex, Node>,

    // count cache hit in batch update
    pub cache_hit_reads: i32,
    pub cache_hit_writes: i32,
    /// Cache of the hashes for the "default" nodes (e.g. ones that are absent in the tree).
    prehashed: Vec<Hash>,
    /// Cache storing the already calculated hashes for nodes
    /// allowing us to avoid calculating the hash of the element more than once.
    /// `RwLock` is required to fulfill the following criteria:
    ///
    /// - Make method `root_hash` immutable (as it's logically immutable).
    /// - Keep the SMT `Sync` (required for the `rayon` parallelism).
    pub cache: RwLock<FnvHashMap<NodeIndex, Hash>>,
}

// Manual implementation of `Clone` is required, since `RwLock` is not `Clone` by default,
// and `Arc` is not a solution (it will lead to the shallow copies, while we need a deep ones).
impl<T, Hash, H> Clone for SparseMerkleTree<T, Hash, H>
where
    T: GetBits + Clone,
    Hash: Clone + Debug,
    H: Hasher<Hash> + Clone,
{
    fn clone(&self) -> Self {
        // let items = self.items.clone();

        let items_data = self.items.read().expect("Read lock").clone();
        let items = RwLock::new(items_data);

        let prehashed = self.prehashed.clone();
        let tree_depth = self.tree_depth;
        let hasher = self.hasher.clone();
        let root = self.root;
        let root_index = self.root_index.clone();

        // let nodes_data = self.nodes.read().expect("Read lock").clone();
        let nodes = self.nodes.clone();
        // let nodes = RwLock::new(nodes_data);

        // let index_node_map_data = self.index_node_map.read().expect("Read lock").clone();

        let index_node_map = self.index_node_map.clone();

        // let index_node_map = RwLock::new(index_node_map_data);

        let cache_data = self.cache.read().expect("Read lock").clone();
        let cache = RwLock::new(cache_data);

        let cache_hit_reads = self.cache_hit_reads.clone();
        let cache_hit_writes = self.cache_hit_writes.clone();

        Self {
            items,
            hasher,
            tree_depth,
            root,
            root_index,
            nodes,
            index_node_map,
            prehashed,
            cache_hit_reads,
            cache_hit_writes,
            cache,
        }
    }
}

/// Merkle Tree branch node.
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct Node {
    pub depth: Depth,
    pub index: NodeIndex,
    left: Option<NodeRef>,
    right: Option<NodeRef>,
    left_index: Option<NodeIndex>,
    right_index: Option<NodeIndex>,
}

/// Child node direction relatively to its parent.
#[derive(Debug, Clone, Copy)]
enum NodeDirection {
    Left,
    Right,
}

impl NodeDirection {
    /// Given the parent index, calculates the child index with respect to the child direction.
    pub fn child_index(self, parent_idx: NodeIndex) -> NodeIndex {
        // Given the parent index N, its child has indices (2*N) and (2*N + 1).
        match self {
            Self::Left => NodeIndex(parent_idx.0 * 2),
            Self::Right => NodeIndex(parent_idx.0 * 2 + 1),
        }
    }

    /// Creates a child node direction basing on its index.
    pub fn from_idx(idx: NodeIndex) -> Self {
        // Left nodes are always even, right nodes are always odd.
        let is_left = (idx.0 & 1) == 0;

        if is_left {
            Self::Left
        } else {
            Self::Right
        }
    }

    /// Depending on the direction, orders the two elements: "primary" and the "secondary".
    /// Direction is assumed to be related to the "primary" element. Thus,
    /// for the `Left` direction, the order is ("primary", "secondary") - "primary" on the left,
    /// for the `Right`, the order is ("secondary", "primary") - "primary" on the right.
    pub fn order_elements<T>(self, primary_el: T, secondary_el: T) -> (T, T) {
        match self {
            Self::Left => (primary_el, secondary_el),
            Self::Right => (secondary_el, primary_el),
        }
    }

    pub fn get_sibling_index(self, parent_idx: NodeIndex) -> NodeIndex {
        // if direction is right, return left sibling node
        // if direction is left, return right sibling node
        match self {
            Self::Left => NodeIndex(parent_idx.0 * 2 + 1),
            Self::Right => NodeIndex(parent_idx.0 * 2),
        }
    }
}

impl<T, Hash, H> SparseMerkleTree<T, Hash, H>
where
    T: GetBits + Default,
    Hash: Clone + Debug,
    H: Hasher<Hash> + Default,
{
    /// Creates a new tree of certain depth (which determines the
    /// capacity of the tree, since the given height will not be
    /// exceeded).
    pub fn new(tree_depth: Depth) -> Self {
        assert!(tree_depth > 1);
        let hasher = H::default();

        // let items = FnvHashMap::default();
        let items = RwLock::new(FnvHashMap::default());

        // let nodes = RwLock::new(Vec::default());
        // nodes.write().expect("write lock").push(Node {
        //     index: NodeIndex(1),
        //     depth: 0,
        //     left: None,
        //     right: None,
        //     left_index: None,
        //     right_index: None,
        // });

        let nodes = vec![Node {
            index: NodeIndex(1),
            depth: 0,
            left: None,
            right: None,
            left_index: None,
            right_index: None,
        }];

        // TODO: memory efficientcy, use reference instead
        // let index_node_map = RwLock::new(FnvHashMap::default());
        // index_node_map.write().expect("write lock").insert(
        //     NodeIndex(1),
        //     Node {
        //         index: NodeIndex(1),
        //         depth: 0,
        //         left: None,
        //         right: None,
        //         left_index: None,
        //         right_index: None,
        //     },
        // );
        let mut index_node_map = FnvHashMap::default();
        index_node_map.insert(
            NodeIndex(1),
            Node {
                index: NodeIndex(1),
                depth: 0,
                left: None,
                right: None,
                left_index: None,
                right_index: None,
            },
        );

        let mut prehashed = Vec::with_capacity(tree_depth);
        let mut cur = hasher.hash_bits(T::default().get_bits_le());
        prehashed.push(cur.clone());
        for i in 0..tree_depth {
            cur = hasher.compress(&cur, &cur, i);
            prehashed.push(cur.clone());
        }
        prehashed.reverse();

        let cache = RwLock::new(FnvHashMap::default());

        Self {
            tree_depth,
            prehashed,
            items,
            hasher,
            nodes,
            index_node_map,
            cache,
            root: 0,
            cache_hit_reads: 0,
            cache_hit_writes: 0,
            root_index: NodeIndex(1),
        }
    }

    /// Roughly calculates the data on the RAM usage for this tree object.
    /// See the [`TreeMemoryUsage`] doc-comments for details.
    pub fn memory_stats(&self) -> TreeMemoryUsage {
        use std::mem::size_of;
        // // For hashmaps, we use both size of keys and values.
        // let items = self.items.capacity() * (size_of::<ItemIndex>() + size_of::<T>());
        let items = 0;
        let nodes = self.nodes.capacity() * size_of::<Node>();
        // let nodes = self.nodes.read().unwrap().capacity() * size_of::<Node>();
        let prehashed = self.prehashed.capacity() * size_of::<Hash>();
        let cache =
            self.cache.read().unwrap().capacity() * (size_of::<NodeIndex>() + size_of::<Hash>());
        let allocated_total = items + nodes + prehashed + cache;

        TreeMemoryUsage {
            items,
            nodes,
            prehashed,
            cache,
            allocated_total,
        }
    }
}

impl<T, Hash, H> SparseMerkleTree<T, Hash, H>
where
    T: GetBits + Default + Sync + Send,
    Hash: Clone + Debug + Sync + Send + Eq,
    H: Hasher<Hash> + Sync + Send,
{
    /// Verifies the given proof for the given element and index.
    pub fn verify_proof(&self, element_index: u32, element: T, proof: Vec<(Hash, bool)>) -> bool {
        let mut proof_index = 0;
        let mut aggregated_hash = self.hasher.hash_bits(element.get_bits_le());
        for (level, (hash, dir)) in proof.into_iter().enumerate() {
            let (lhs, rhs) = if dir {
                proof_index |= 1 << level;
                (hash, aggregated_hash)
            } else {
                (aggregated_hash, hash)
            };

            aggregated_hash = self.hasher.compress(&lhs, &rhs, level);
        }
        proof_index == element_index && aggregated_hash == self.root_hash()
    }
}

impl<T, Hash, H> SparseMerkleTree<T, Hash, H>
where
    T: GetBits + Default + Sync + Send,
    Hash: Clone + Debug + Sync + Send,
    H: Hasher<Hash> + Sync + Send,
{
    const ROOT_ITEM_IDX: NodeRef = 0;

    /// Inserts an element to the tree.
    pub fn insert(&mut self, item_index: u32, item: T) {
        let item_index = item_index as ItemIndex;

        assert!(item_index < self.capacity());
        let tree_depth = self.tree_depth;
        let leaf_index: NodeIndex = NodeIndex((1 << tree_depth) + item_index);

        self.items
            .write()
            .expect("write lock")
            .insert(item_index, item);
        // self.items.insert(item_index, item);

        // Invalidate the root cache.
        self.cache
            .write()
            .expect("write lock")
            .remove(&NodeIndex(1));
        self.cache_hit_writes += 1;

        // Traverse the tree, starting from the root.
        // Since our tree is "sparse", it can have gaps.
        // Essentially this means that we should go down from the root node, calculating
        // the expected direction, and find the node which does not have a child with this direction,
        // and insert it.
        //
        // Schematic representation:
        //
        // ```text
        //     __(1)__
        //    |       |
        //  _(2)_    (3)_
        // |     |       |
        // A     B       D
        // ```
        //
        // 1 - Root node.
        // 2 - Node with both left and right children.
        // 3 - Node with only the right children.
        //
        // If we want to insert value C to the third position, we will start from (1), then go to the (3)
        // and there insert the value as the left child:
        //
        // ```text
        //     __(1)__
        //    |       |
        //  _(2)_   _(3)_
        // |     | |     |
        // A     B C     D
        // ```
        let mut current_node_ref = self.root;

        // let mut current_node_index = self.root_index;

        loop {
            let current_node = self.nodes[current_node_ref].clone();

            let current_level = self.calculate_level(current_node.depth);

            // We have the index of the child, and since at every level the index is
            // divided by 2, to check the direction at some level we may just check
            // the corresponding bit in the child index.
            // Even value will mean the "left" direction, and the odd one will mean "right".
            let going_right = (leaf_index.0 & (1 << current_level)) > 0;
            let (dir, child_ref) = if going_right {
                // let (dir, child_index) = if going_right {
                (NodeDirection::Right, current_node.right)
            } else {
                (NodeDirection::Left, current_node.left)
            };
            // if next ref exists, for indexed, would be child index
            if let Some(next_ref) = child_ref {
                let next = self.nodes[next_ref].clone();
                let leaf_index_normalized = NodeIndex(leaf_index.0 >> (tree_depth - next.depth));

                // Check if the `next` node is the node we should update.
                if leaf_index_normalized == next.index {
                    // Yep, we should update the `next` node.

                    // Start from invalidating the cache for this node.
                    self.wipe_cache(next.index, current_node.index);

                    // We should go at least one full level deeper.
                    if next.index == leaf_index {
                        // We reached the leaf, no further updating required.
                        // All the outdated caches are invalidated, and the leaf value
                        // was inserted below.
                        break;
                    } else {
                        // We didn't reach the leaf layer, thus we should keep going down the tree.
                        current_node_ref = next_ref;
                        // current_node_index = next.index;
                        continue;
                    }
                } else {
                    // Next node is **not** the node we must update.
                    // We have to insert one additional node which will have the
                    // `next` node and our node as children.

                    // Find the intersection point: the biggest index which will
                    // be the parent for both of the nodes.
                    let common_parent_index = {
                        let mut first_node_idx = leaf_index_normalized;
                        let mut second_node_idx = next.index;

                        // As the index of the parent to the node can be calculated
                        // by dividing it by two, we keep dividing both indices until
                        // they are equal. Once they are equal, we've got the common parent index
                        while first_node_idx != second_node_idx {
                            first_node_idx.0 >>= 1;
                            second_node_idx.0 >>= 1;
                        }
                        first_node_idx
                    };

                    // Invalidate the cache for the intersection point.
                    self.wipe_cache(common_parent_index, current_node.index);

                    // Insert the leaf node.
                    let leaf_ref = self.insert_node(leaf_index, tree_depth, None, None, None, None);

                    // Find the direction of our node relatively to the parent
                    // and order "our" node, then order the references to match the directions.
                    let direction = if leaf_index_normalized > next.index {
                        NodeDirection::Right
                    } else {
                        NodeDirection::Left
                    };

                    let (lhs_ref, rhs_ref) =
                        direction.order_elements(Some(leaf_ref), Some(next_ref));

                    // Insert a split node and set it as a child for the current node.
                    let split_depth = Self::depth(common_parent_index);

                    let child_index = direction.child_index(common_parent_index);

                    // sibling index
                    let sibling_index = direction.get_sibling_index(common_parent_index);
                    let (lhs_index, rhs_index) =
                        direction.order_elements(Some(child_index), Some(sibling_index));

                    let split_node_ref = self.insert_node(
                        common_parent_index,
                        split_depth,
                        lhs_ref,
                        rhs_ref,
                        lhs_index,
                        rhs_index,
                    );
                    self.add_child_node(current_node_ref, dir, split_node_ref, child_index);
                    break;
                }
            } else {
                // There is no child within the direction of the node to insert.
                // We must simply insert the leaf and make it a child of the latest
                // existing parent node.
                // No further processing is required.

                let leaf_ref = self.insert_node(leaf_index, tree_depth, None, None, None, None);
                self.add_child_node(current_node_ref, dir, leaf_ref, leaf_index);
                break;
            }
        }
    }

    /// Removes an element with a given index, and returns the removed
    /// element (if it existed in the tree).
    pub fn remove(&mut self, index: u32) -> Option<T> {
        let index = index as ItemIndex;

        // remove leaf
        let old = self.items.write().expect("write lock").remove(&index);
        // let old = self.items.remove(&index);
        let item = T::default();

        // insert default node
        self.insert(index as u32, item);

        old
    }

    /// Returns the Merkle root hash of the tree. This operation can cost up to O(N*logN):
    /// the root hash is calculated in this method, and it will build the whole hash tree
    /// if this method was not called. The intermediate calculation results are caches though,
    /// thus follow-up invocations will cost less.
    pub fn root_hash(&self) -> Hash {
        let (root_hash, intermediate_hashes) = self.get_hash(Self::ROOT_ITEM_IDX);

        // println!(
        //     "Intermediate hashes length: {:?}",
        //     intermediate_hashes.len()
        // );

        // Store all the intermediate hashes in the cache.
        for (item_idx, hash) in intermediate_hashes {
            self.cache
                .write()
                .expect("write lock")
                .insert(item_idx, hash);
            // self.cache_hit_writes += 1;
        }
        root_hash
    }

    /// Returns the capacity of the tree (how many items can the tree hold).
    pub fn capacity(&self) -> u64 {
        1 << self.tree_depth
    }

    /// Creates a proof of existence for a certain element of the tree.
    /// Returned value is a list of pairs, where the first element is
    /// the aggregated coupling hash for current layer, and the second is
    /// the direction.
    pub fn merkle_path(&mut self, index: u32) -> Vec<(Hash, bool)> {
        assert!((index as ItemIndex) < self.capacity());

        // By calculating the root hash we update the cache of hashes,
        // which we will use to build a proof.
        // After updating the cache, there will be no "unknown" hashes:
        // the required hash will either be in hash, or it will be a precomputed
        // hash for the current depth.
        let _root_hash = self.root_hash();

        let index = index as ItemIndex;

        // Node indexes use an additional bit set at the position `(1 << self.tree_depth)`
        // for indexing.
        let mut cur_index: NodeIndex = NodeIndex((1 << self.tree_depth) + index);
        let mut proof = Vec::new();

        // We will access cache at the every iteration of our cycle, thus obtaining
        // the lock here and passing it as an argument is more efficient than
        // obtaining it for every iteration separately.
        let cache_lock = self.cache.read().expect("Read lock");

        // We go through all the depths starting from `tree_depth` to collect the proof
        // hashes.
        for depth in (1..=self.tree_depth).rev() {
            let (neighbor_hash, dir) = self.get_calculated_node_hash(&cache_lock, depth, cur_index);
            // At each step we go one height closer to the root and replacing the
            // current node index with the index of its parent.
            // For node with an index `N` the index of the parent will always be `N / 2`.
            cur_index.0 >>= 1;

            proof.push((neighbor_hash, dir));
        }

        proof
    }

    /// A helper method for `merkle_path`: obtains the hash for the node with provided index,
    /// assuming that it is already calculated. That is, if the node is absent in the cache,
    /// it is assumed that it has the precomputed hash for the current depth.
    fn get_calculated_node_hash(
        &self,
        cache_lock: &RwLockReadGuard<FnvHashMap<NodeIndex, Hash>>,
        depth: usize,
        node_index: NodeIndex,
    ) -> (Hash, bool) {
        // By `xor`ing the node index with 1 we will obtain the index of the neighbor node:
        // if the index of current node is (2 * N), it will become (2 * N + 1), and vice versa:
        // for (2 * N + 1) it will become (2 * N).
        let neighbor_index = NodeIndex(node_index.0 ^ 1);

        // `node_index` is the index of the current node according to the current depth,
        // thus LSB represents the direction for the current depth.
        let going_right = (node_index.0 & 1) != 0;

        // If hash is not presented in the cache, it must be a precomputed one.
        let neighbor_hash = match cache_lock.get(&neighbor_index) {
            Some(hash) => {
                // self.cache_hit_reads += 1;
                hash.clone()
            }
            None => self.prehashed[depth].clone(),
        };
        (neighbor_hash, going_right)
    }

    /// Calculates the depth ("layer") of the element with the provided index.
    fn depth(index: NodeIndex) -> Depth {
        let mut level: Depth = 0;
        let mut i = index;
        while i.0 > 1 {
            level += 1;
            i.0 >>= 1;
        }
        level
    }

    // Returns the *hash* capacity of the tree (how many hashes can the tree hold)
    #[allow(dead_code)]
    fn nodes_capacity(&self) -> usize {
        (1 << (self.tree_depth + 1)) - 1
    }

    /// Removes the entry with provided index from the hashes cache, as well
    /// as its parent entries, limited by the `parent` index.
    fn wipe_cache(&mut self, child: NodeIndex, parent: NodeIndex) {
        let mut cache = self.cache.write().expect("write lock");
        if cache.remove(&child).is_some() {
            // Item existed in cache, now we should go up the tree
            // and remove parent hashes, until we reach the provided
            // `parent` index.
            let mut i = NodeIndex(child.0 >> 1);
            while i > parent {
                cache.remove(&i);
                self.cache_hit_writes += 1;
                i.0 >>= 1;
            }
        }
    }

    /// Inserts the node to the tree and returns it's position.
    fn insert_node(
        &mut self,
        index: NodeIndex,
        depth: Depth,
        left: Option<NodeRef>,
        right: Option<NodeRef>,
        left_index: Option<NodeIndex>,
        right_index: Option<NodeIndex>,
    ) -> NodeRef {
        // self.nodes.write().expect("write lock").push(Node {
        //     depth,
        //     index,
        //     left,
        //     right,
        //     left_index: left_index,
        //     right_index: right_index,
        // });

        self.nodes.push(Node {
            depth,
            index,
            left,
            right,
            left_index: left_index,
            right_index: right_index,
        });

        // self.index_node_map.write().expect("write lock").insert(
        //     index,
        //     Node {
        //         depth,
        //         index,
        //         left,
        //         right,
        //         left_index: left_index,
        //         right_index: right_index,
        //     },
        // );

        self.index_node_map.insert(
            index,
            Node {
                depth,
                index,
                left,
                right,
                left_index: left_index,
                right_index: right_index,
            },
        );

        // self.nodes.read().expect("read lock").len() - 1

        self.nodes.len() - 1
    }

    /// Sets a child node for an existing node in the tree.
    fn add_child_node(
        &mut self,
        node_ref: NodeRef,
        dir: NodeDirection,
        child_ref: NodeRef,
        child_index: NodeIndex,
    ) {
        // let node = &mut self.nodes.write().expect("write lock")[node_ref];
        let node = &mut self.nodes[node_ref];

        match dir {
            NodeDirection::Left => {
                node.left = Some(child_ref);
                node.left_index = Some(child_index)
            }
            NodeDirection::Right => {
                node.right = Some(child_ref);
                node.right_index = Some(child_index)
            }
        }
    }

    /// Finds the hash of the node's child, using one of the following strategy:
    /// - If the hash exists in cache, the cached value is returned;
    /// - If the element with the child's index absents in the tree, the precomputed hash
    ///   for the corresponding layer is returned.
    /// - Otherwise, the hash for the child is actually calculated using `calculate_child_hash`
    ///   method.
    fn get_child_hash(&self, parent: &Node, dir: NodeDirection) -> (Hash, Vec<(NodeIndex, Hash)>) {
        let child_ref = match dir {
            NodeDirection::Left => parent.left,
            NodeDirection::Right => parent.right,
        };

        let child_index = dir.child_index(parent.index);

        // Check if the child data exists in the cache.
        if let Some(cached) = self.cache.read().expect("Read lock").get(&child_index) {
            // Cache hit, no calculations required.
            let updates = vec![];

            (cached.clone(), updates)
        } else {
            match child_ref {
                Some(child_ref) => {
                    // Child exists in the tree, we must calculate the underlying hashes.
                    self.calculate_child_hash(child_ref, parent)
                }
                None => {
                    let default_hash_for_layer = self.prehashed[parent.depth + 1].clone();
                    let updates = vec![];
                    (default_hash_for_layer, updates)
                }
            }
        }
    }

    /// Calculates the hash of the node's child given the parent node and the child direction.
    fn calculate_child_hash(
        &self,
        child_ref: NodeRef,
        parent: &Node,
    ) -> (Hash, Vec<(NodeIndex, Hash)>) {
        // let child = &self.nodes.write().expect("write lock")[child_ref];

        let child = &self.nodes[child_ref];

        // Get the hash of the child itself.
        let (mut cur_hash, mut updates) = self.get_hash(child_ref);

        // Now, we should fill the layer "gaps" between child and parent.
        // This means that we should go through layers of the child and parent,
        // and update the obtained hash with the precomputed hash for this layer.
        let mut cur_depth = child.depth - 1;
        let mut cur_idx = child.index;

        // The topmost layer has depth 0, so we go from the higher layer to the lower one.
        while cur_depth > parent.depth {
            // Before combining current hash with the precomputed one, we should determine the order
            // (basically, the position of "our" hash relatively to the next-layer parent).
            let direction = NodeDirection::from_idx(cur_idx);

            let supplement_hash = self.prehashed[cur_depth + 1].clone();
            let (lhs_hash, rhs_hash) = direction.order_elements(cur_hash, supplement_hash);

            cur_hash = self.calculate_hash(cur_depth, &lhs_hash, &rhs_hash);

            // At each iteration our index become 2 times smaller, and the depth is decremented by 1.
            cur_depth -= 1;
            cur_idx.0 >>= 1;

            //self.cache.insert(cur_idx, cur_hash.clone());
            updates.push((cur_idx, cur_hash.clone()));
        }
        (cur_hash, updates)
    }

    /// Calculates the tree hash for the element given its position.
    /// Returns the calculates hash and the list of updated underlying
    /// hashes together with their positions.
    fn get_hash(&self, node_ref: NodeRef) -> (Hash, Vec<(NodeIndex, Hash)>) {
        // let node = &self.nodes.write().expect("write lock")[node_ref].clone();
        let node = &self.nodes[node_ref].clone();

        // if (node.index == 4) {
        //     println!("node.index: {:?}", node.index);
        // }

        // Calculate the hash of this node, and collect the underlying updates.
        // The updates list won't contain the current node, we will add it below.
        let (hash, mut updates) = {
            if node.depth == self.tree_depth {
                // leaf node: return item hash
                let item_index: ItemIndex = (node.index.0 - (1 << self.tree_depth)) as ItemIndex;

                // let item_bits = self.items[&item_index].get_bits_le();
                // let item_hash = self.hasher.hash_bits(item_bits);
                let item_lock = self.items.read().expect("Read lock");
                let item = item_lock.get(&item_index);
                let item_bits = item.as_ref().expect("Item is None").get_bits_le();
                let item_hash = self.hasher.hash_bits(item_bits);

                // There are no underlying updates for leaf node.
                let updates = vec![];

                (item_hash, updates)
            } else {
                // Not a leaf node: recursively calculate the hashes up to this node.

                // Use `rayon` to calculate hashes in parallel.
                let (left_hashes, right_hashes) = rayon::join(
                    || self.get_child_hash(node, NodeDirection::Left),
                    || self.get_child_hash(node, NodeDirection::Right),
                );

                // let left_hashes = self.get_child_hash(node, NodeDirection::Left);
                // let right_hashes = self.get_child_hash(node, NodeDirection::Right);

                // let (left_hashes, right_hashes) = rayon::join(
                //     || self.get_child_hash(node, NodeDirection::Left),
                //     || self.get_child_hash(node, NodeDirection::Right),
                // );

                let (lhs_hash, lhs_updates) = left_hashes;
                let (rhs_hash, rhs_updates) = right_hashes;

                let hash = self.calculate_hash(node.depth, &lhs_hash, &rhs_hash);

                // Merge left and right updates.
                let mut updates = lhs_updates;
                updates.extend(rhs_updates);
                (hash, updates)
            }
        };

        // Add the current node hash to the list of updates.
        updates.push((node.index, hash.clone()));

        // println!(
        //     // "Normal insert: node {}, difference: {}",
        //     node.index.0,
        //     start_hash.elapsed().as_millis(),
        // );
        //self.cache.insert(node.index, hash.clone());
        (hash, updates)
    }

    fn calculate_hash(&self, cur_depth: usize, lhs_hash: &Hash, rhs_hash: &Hash) -> Hash {
        // Level is used by hasher for personalization
        let level = self.calculate_level(cur_depth);

        // Calculate the hash of this node.
        self.hasher.compress(lhs_hash, rhs_hash, level)
    }

    fn calculate_level(&self, cur_depth: usize) -> usize {
        self.tree_depth - cur_depth - 1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct SparseMerkleTreeSerializableCacheBN256<T: GetBits + Default + 'static> {
    root: NodeRef,
    nodes: Vec<Node>,
    items: Vec<(ItemIndex, T)>,
    cache: Vec<(NodeIndex, [u8; 32])>,
    index_node_map: Vec<(NodeIndex, Node)>,
}

impl<T> SparseMerkleTreeSerializableCacheBN256<T>
where
    T: GetBits + Default + Decode + Clone + bincode::Encode,
{
    pub fn encode_bincode(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, bincode::config::standard())
            .expect("Unable to encode Merkle Tree cache")
    }

    pub fn decode_bincode(data: &[u8]) -> Self {
        bincode::decode_from_slice(data, bincode::config::standard())
            .expect("Unable to decode Merkle Tree cache")
            .0
    }
}

impl<T, H> SparseMerkleTree<T, Fr, H>
where
    T: GetBits + Default + Clone,
    H: Hasher<Fr>,
{
    pub fn get_internals(&self) -> SparseMerkleTreeSerializableCacheBN256<T> {
        SparseMerkleTreeSerializableCacheBN256 {
            root: self.root,
            // nodes: self.nodes.read().unwrap().clone(),
            nodes: self.nodes.clone(),
            index_node_map: self
                .index_node_map
                .iter()
                .map(|(k, node)| (*k, (*node).clone()))
                .collect(),
            items: self
                .items
                .read()
                .unwrap()
                .iter()
                .map(|(idx, node)| (*idx, (*node).clone()))
                .collect(),
            cache: self
                .cache
                .read()
                .unwrap()
                .iter()
                .map(|(idx, fr)| {
                    let mut fr_bytes = [0u8; 32];
                    fr.into_repr()
                        .write_be(&mut fr_bytes[..])
                        .expect("Fr write error");
                    (*idx, fr_bytes)
                })
                .collect(),
        }
    }

    pub fn set_internals(&mut self, internals: SparseMerkleTreeSerializableCacheBN256<T>) {
        self.root = internals.root;
        // self.nodes = RwLock::new(internals.nodes);
        self.nodes = internals.nodes;
        self.items = RwLock::new(
            internals
                .items
                .into_iter()
                .map(|(idx, node)| (idx, node))
                .collect(),
        );
        self.cache = RwLock::new(
            internals
                .cache
                .into_iter()
                .map(|(idx, fr_bytes)| {
                    let mut fr_repr = <Fr as PrimeField>::Repr::default();
                    fr_repr.read_be(&fr_bytes[..]).expect("Fr read error");
                    (idx, Fr::from_repr(fr_repr).expect("Fr decode error"))
                })
                .collect(),
        );
        self.index_node_map = internals.index_node_map.into_iter().collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, Clone)]
    struct TestHasher;

    #[derive(Debug, PartialEq, Default, Clone, Copy)]
    struct TestLeaf(u64);

    impl GetBits for TestLeaf {
        fn get_bits_le(&self) -> Vec<bool> {
            let mut acc = Vec::new();
            let mut i = self.0 + 1;
            for _ in 0..16 {
                acc.push(i & 1 == 1);
                i >>= 1;
            }
            acc
        }
    }

    impl Hasher<u64> for TestHasher {
        fn hash_bits<I: IntoIterator<Item = bool>>(&self, value: I) -> u64 {
            let mut acc = 0;
            let v: Vec<bool> = value.into_iter().collect();
            for i in v.iter() {
                acc <<= 1;
                if *i {
                    acc |= 1
                };
            }
            acc
        }

        fn hash_elements<I: IntoIterator<Item = u64>>(&self, _elements: I) -> u64 {
            unreachable!("Hash elements is specific for rescue hash and used only for pubkey hash derivation")
        }

        fn compress(&self, lhs: &u64, rhs: &u64, i: usize) -> u64 {
            (11 * lhs + 17 * rhs + 1 + i as u64) % 1_234_567_891
            // deg!("compress {} {}, {} => {}", lhs, rhs, i, r);
        }
    }

    type TestSMT = SparseMerkleTree<TestLeaf, u64, TestHasher>;

    #[test]
    fn test_merkle_tree_insert() {
        let mut tree = TestSMT::new(3);

        assert_eq!(tree.capacity(), 8);

        tree.insert(0, TestLeaf(1));
        assert_eq!(tree.root_hash(), 697_516_875);

        tree.insert(0, TestLeaf(2));
        assert_eq!(tree.root_hash(), 741_131_083);

        tree.insert(3, TestLeaf(2));
        assert_eq!(tree.root_hash(), 793_215_819);
    }

    /// Performs some basic insert/remove operations.
    #[test]
    fn merkle_tree_workflow() {
        let mut tree = TestSMT::new(3);

        // Add one element with known-before hash.
        tree.insert(0, TestLeaf(1));
        assert_eq!(tree.root_hash(), 697_516_875);

        // Add more elements.
        for idx in 1..8 {
            tree.insert(idx, TestLeaf(idx as u64));
        }

        // Remove them (and check that within removing we can obtain them).
        for idx in (1..8).rev() {
            assert_eq!(tree.remove(idx), Some(TestLeaf(idx as u64)));
        }

        // The first element left only, hash should be the same as in the beginning.
        assert_eq!(tree.root_hash(), 697_516_875);
    }

    /// Checks the correctness of the built Merkle proofs
    #[test]
    fn merkle_path_test() {
        // Test vector holds pairs (index, value).
        let test_vector = [(0, 2), (3, 2)];
        // Precomputed root hash for the test vector above.
        let expected_root_hash = 793_215_819;

        // Create the tree and fill it with values.
        let mut tree = TestSMT::new(3);
        assert_eq!(tree.capacity(), 8);
        for &(idx, value) in &test_vector {
            tree.insert(idx, TestLeaf(value));
        }
        assert_eq!(tree.root_hash(), expected_root_hash);

        // Check the proof for every element.
        for &(idx, value) in &test_vector[..] {
            let merkle_proof = tree.merkle_path(idx);

            let hasher = TestHasher::default();

            // To check the proof, we fold it starting from the hash of the value
            // and updating with the hashes from the proof.
            // We should obtain the root hash at the end if the proof is correct.
            let mut level = 0;
            let mut proof_index: ItemIndex = 0;
            let mut aggregated_hash = hasher.hash_bits(value.get_bits_le());
            for (hash, dir) in merkle_proof {
                let (lhs, rhs) = if dir {
                    proof_index |= 1 << level;
                    (hash, aggregated_hash)
                } else {
                    (aggregated_hash, hash)
                };

                aggregated_hash = hasher.compress(&lhs, &rhs, level);

                level += 1;
            }

            assert_eq!(level, tree.tree_depth);
            assert_eq!(proof_index, idx as u64);
            assert_eq!(aggregated_hash, 793_215_819);
        }
        // Since sparse merkle tree is by default "filled" with default values,
        // we can check the proofs for elements which we did not insert by ourselves.
        // Given the tree depth 3, the tree capacity is 8 (2^3).
        let absent_elements = [1, 2, 4, 5, 6, 7];
        let default_value = 0;

        for &idx in &absent_elements {
            let merkle_proof = tree.merkle_path(idx);

            let hasher = TestHasher::default();

            // To check the proof, we fold it starting from the hash of the value
            // and updating with the hashes from the proof.
            // We should obtain the root hash at the end if the proof is correct.
            let mut level = 0;
            let mut proof_index: ItemIndex = 0;
            let mut aggregated_hash = hasher.hash_bits(default_value.get_bits_le());
            for (hash, dir) in merkle_proof {
                let (lhs, rhs) = if dir {
                    proof_index |= 1 << level;
                    (hash, aggregated_hash)
                } else {
                    (aggregated_hash, hash)
                };

                aggregated_hash = hasher.compress(&lhs, &rhs, level as usize);

                level += 1;
            }

            assert_eq!(level, tree.tree_depth);
            assert_eq!(proof_index, idx as u64);
            assert_eq!(aggregated_hash, 793_215_819);
        }
    }

    #[test]
    fn merkle_tree_batch_insert_empty() {
        let mut tree_1 = TestSMT::new(3);

        // Add more elements.
        for idx in 0..8 {
            tree_1.insert(idx, TestLeaf(idx as u64));
        }
        // updated node 0 value to 1
        let tree1_hash = tree_1.root_hash();

        // set up second
        let mut tree_2 = TestSMT::new(3);
        for idx in 0..8 {
            tree_2.insert(idx, TestLeaf(idx as u64));
        }
        // cache hash
        tree_2.root_hash();

        let updates: Vec<(ItemIndex, SMTLeafUpdate<TestLeaf>)> = vec![];
        // Act
        let tree2_result = tree_2.batch_insert_parallel(updates);
        // let tree2_hash = tree_2.root_hash();

        let tree2_hash = tree2_result.unwrap();

        // Assert
        assert_eq!(tree1_hash, tree2_hash);
    }

    /// Performs batch update insert
    #[test]
    fn merkle_tree_batch_insert_single() {
        // Arrange
        let mut tree_1 = TestSMT::new(3);
        let mut tree_2 = TestSMT::new(3);

        // Add more elements.
        for idx in 0..3 {
            tree_1.insert(idx, TestLeaf(idx as u64));
            tree_2.insert(idx, TestLeaf(idx as u64));
        }
        // cache trees
        tree_1.root_hash();
        tree_2.root_hash();

        // Act
        // update tree 1
        tree_1.insert(2, TestLeaf(2));

        // // Create update in tree 2
        let updates: SMTLeafUpdates<TestLeaf> = vec![(
            2,
            SMTLeafUpdate::Insert {
                item_index: (2),
                item: TestLeaf(2 as u64),
            },
        )];

        // Act
        let tree2_result = tree_2.batch_insert(updates);

        let tree1_hash_final = tree_1.root_hash();
        let tree2_hash_final = tree2_result.unwrap();

        // // Assert
        println!(
            "hash 1: {:?}, hash 2: {:?}",
            tree1_hash_final, tree2_hash_final
        );

        assert_eq!(tree1_hash_final, tree2_hash_final);
    }

    /// Performs batch update insert
    #[test]
    fn merkle_tree_batch_insert_multiple() {
        let mut tree_1 = TestSMT::new(2);

        // Add more elements.
        for idx in 0..4 {
            tree_1.insert(idx, TestLeaf(idx as u64));
        }
        // updated node 0 value to 1
        tree_1.insert(0, TestLeaf(1));
        tree_1.insert(1, TestLeaf(2));
        tree_1.insert(2, TestLeaf(3));

        let tree1_hash = tree_1.root_hash();

        // set up second
        let mut tree_2 = tree_1.clone();

        let updates: Vec<(ItemIndex, SMTLeafUpdate<TestLeaf>)> = vec![
            (
                0,
                SMTLeafUpdate::Insert {
                    item_index: (0),
                    item: TestLeaf(1),
                },
            ),
            (
                1,
                SMTLeafUpdate::InsertNew {
                    item_index: (1),
                    item: TestLeaf(2),
                },
            ),
            (
                2,
                SMTLeafUpdate::InsertNew {
                    item_index: (2),
                    item: TestLeaf(3),
                },
            ),
        ];

        // Act
        let tree2_result = tree_2.batch_insert(updates);
        // let tree2_hash = tree_2.root_hash();

        let tree2_hash = tree2_result.unwrap();

        // Assert
        println!("hash 1: {:?}, hash 2: {:?}", tree1_hash, tree2_hash);

        assert_eq!(tree1_hash, tree2_hash);
    }

    #[test]
    fn merkle_tree_batch_insert_multiple_remove() {
        let mut tree_1 = TestSMT::new(3);

        tree_1.insert(0, TestLeaf(1));
        tree_1.insert(2, TestLeaf(3));
        tree_1.insert(4, TestLeaf(5));

        let tree1_hash = tree_1.root_hash();

        let mut tree_2 = TestSMT::new(3);

        let updates: Vec<(ItemIndex, SMTLeafUpdate<TestLeaf>)> = vec![
            (
                0,
                SMTLeafUpdate::InsertNew {
                    item_index: (0),
                    item: TestLeaf(1),
                },
            ),
            (
                2,
                SMTLeafUpdate::InsertNew {
                    item_index: (2),
                    item: TestLeaf(3),
                },
            ),
            (
                4,
                SMTLeafUpdate::InsertNew {
                    item_index: (4),
                    item: TestLeaf(5),
                },
            ),
        ];

        // Act
        let tree2_result = tree_2.batch_insert(updates).unwrap();

        // Assert
        assert_eq!(tree1_hash, tree2_result);
    }
}
