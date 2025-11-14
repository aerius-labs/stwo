//! Metal quotient operations (QuotientOps trait implementation).
//!
//! This module implements quotient accumulation with GPU acceleration for large workloads
//! and SIMD fallback for small workloads.

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::core::poly::circle::CircleDomain;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::QuotientOps;

use super::MetalBackend;

impl QuotientOps for MetalBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        // TODO(Phase 3): Dispatch to Metal GPU for large quotient accumulation
        // Transmute column slice to SIMD backend types (layout-compatible)
        let simd_columns: &[&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>] =
            unsafe { &*(columns as *const _ as *const _) };
        let simd_result = SimdBackend::accumulate_quotients(
            domain,
            simd_columns,
            random_coeff,
            sample_batches,
            log_blowup_factor,
        );

        // Transmute result back
        unsafe { std::mem::transmute(simd_result) }
    }
}
