//! Metal FRI operations (FriOps trait implementation).
//!
//! This module implements FRI folding operations with GPU acceleration for large workloads
//! and SIMD fallback for small workloads.

use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::fri::FriOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;

use super::MetalBackend;

impl FriOps for MetalBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        // TODO(Phase 2): Dispatch to Metal GPU for large FRI folding
        // Transmute to SIMD types (layout-compatible, same column types)
        let simd_eval: &LineEvaluation<SimdBackend> =
            unsafe { &*(eval as *const _ as *const _) };
        let simd_twiddles: &TwiddleTree<SimdBackend> =
            unsafe { &*(twiddles as *const _ as *const _) };
        let simd_result = SimdBackend::fold_line(simd_eval, alpha, simd_twiddles);

        // Convert back by transmuting the result (layout-compatible)
        let metal_result: LineEvaluation<MetalBackend> =
            unsafe { std::mem::transmute(simd_result) };
        metal_result
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        // TODO(Phase 2): Dispatch to Metal GPU for large circle-to-line folding
        // Transmute to SIMD types
        let simd_dst: &mut LineEvaluation<SimdBackend> =
            unsafe { &mut *(dst as *mut _ as *mut _) };
        let simd_src: &SecureEvaluation<SimdBackend, BitReversedOrder> =
            unsafe { &*(src as *const _ as *const _) };
        let simd_twiddles: &TwiddleTree<SimdBackend> =
            unsafe { &*(twiddles as *const _ as *const _) };
        SimdBackend::fold_circle_into_line(simd_dst, simd_src, alpha, simd_twiddles)
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        // Decomposition uses SIMD
        let simd_eval: &SecureEvaluation<SimdBackend, BitReversedOrder> =
            unsafe { &*(eval as *const _ as *const _) };
        let (simd_result, field) = SimdBackend::decompose(simd_eval);
        // Transmute result back
        let metal_result: SecureEvaluation<MetalBackend, BitReversedOrder> =
            unsafe { std::mem::transmute(simd_result) };
        (metal_result, field)
    }
}
