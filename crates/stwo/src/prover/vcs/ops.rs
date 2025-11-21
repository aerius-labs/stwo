use serde::{Deserialize, Serialize};

use crate::core::fields::m31::BaseField;
use crate::core::vcs::MerkleHasher;
use crate::prover::backend::{Col, Column, ColumnOps};

/// Trait for performing Merkle operations on a commitment scheme.
pub trait MerkleOps<H: MerkleHasher>:
    ColumnOps<BaseField> + ColumnOps<H::Hash> + for<'de> Deserialize<'de> + Serialize
{
    /// Commits on an entire layer of the Merkle tree.
    /// See [MerkleHasher] for more details.
    ///
    /// The layer has 2^`log_size` nodes that need to be hashed. The topmost layer has 1 node,
    /// which is a hash of 2 children and some columns.
    ///
    /// `prev_layer` is the previous layer of the Merkle tree, if this is not the leaf layer.
    /// That layer is assumed to have 2^(`log_size`+1) nodes.
    ///
    /// `columns` are the extra columns that need to be hashed in each node.
    /// They are assumed to be of size 2^`log_size`.
    ///
    /// Returns the next Merkle layer hashes.
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, H::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, H::Hash>;

    /// Batched commit for multiple consecutive node-only layers (no columns).
    /// This allows backends to optimize by submitting all layers in a single GPU command buffer.
    ///
    /// `initial_layer` is the starting layer (previous layer for the first new layer).
    /// `num_layers` is the number of node layers to compute.
    ///
    /// Returns all computed layers (length = num_layers).
    ///
    /// Default implementation falls back to layer-by-layer commit_on_layer.
    fn commit_node_layers_batched(
        initial_layer: &Col<Self, H::Hash>,
        num_layers: u32,
    ) -> Vec<Col<Self, H::Hash>> {
        let mut layers = Vec::new();
        let mut prev_layer = initial_layer;
        let mut log_size = initial_layer.len().ilog2() - 1;

        for _ in 0..num_layers {
            let layer = Self::commit_on_layer(log_size, Some(prev_layer), &[]);
            layers.push(layer);
            prev_layer = layers.last().unwrap();
            log_size = log_size.saturating_sub(1);
        }

        layers
    }
}
