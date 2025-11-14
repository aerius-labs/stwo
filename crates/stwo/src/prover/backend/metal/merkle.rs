//! Metal Merkle operations (MerkleOps trait implementation).
//!
//! This module implements Merkle tree construction with GPU acceleration for large workloads
//! and SIMD fallback for small workloads.

use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleHasher, Blake2sMerkleHasher};
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::{Col, ColumnOps};
use crate::prover::vcs::ops::MerkleOps;

use super::MetalBackend;

// Blake2s hash columns are just Vec<Blake2sHash> like in SIMD backend
impl ColumnOps<Blake2sHash> for MetalBackend {
    type Column = Vec<Blake2sHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for MetalBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Blake2sHash> {
        // TODO(Phase 3): Dispatch to Metal GPU for large Merkle operations
        // Transmute columns to SIMD backend types (layout-compatible)
        let simd_columns: &[&Col<SimdBackend, BaseField>] =
            unsafe { &*(columns as *const _ as *const _) };
        <SimdBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size,
            prev_layer,
            simd_columns,
        )
    }
}

impl MerkleOps<Blake2sM31MerkleHasher> for MetalBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Blake2sHash> {
        // TODO(Phase 3): Dispatch to Metal GPU for large Merkle operations
        // Transmute columns to SIMD backend types (layout-compatible)
        let simd_columns: &[&Col<SimdBackend, BaseField>] =
            unsafe { &*(columns as *const _ as *const _) };
        <SimdBackend as MerkleOps<Blake2sM31MerkleHasher>>::commit_on_layer(
            log_size,
            prev_layer,
            simd_columns,
        )
    }
}
