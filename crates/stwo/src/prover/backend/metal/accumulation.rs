//! Metal accumulation operations (AccumulationOps trait implementation).
//!
//! This module implements field accumulation operations, delegating to SIMD for now.

use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::AccumulationOps;

use super::MetalBackend;

impl AccumulationOps for MetalBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        // Accumulation is typically small, transmute and use SIMD
        let simd_column: &mut SecureColumnByCoords<SimdBackend> =
            unsafe { &mut *(column as *mut _ as *mut _) };
        let simd_other: &SecureColumnByCoords<SimdBackend> =
            unsafe { &*(other as *const _ as *const _) };
        SimdBackend::accumulate(simd_column, simd_other)
    }

    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        // Power generation uses SIMD (no backend-specific types)
        SimdBackend::generate_secure_powers(felt, n_powers)
    }
}
