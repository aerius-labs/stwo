//! Metal GKR and MLE operations (GkrOps and MleOps trait implementations).
//!
//! This module implements GKR (Grand product and lookup) operations with MLE support.

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::lookups::gkr_prover::{GkrMultivariatePolyOracle, GkrOps, Layer};
use crate::prover::lookups::mle::{Mle, MleOps};
use crate::prover::lookups::utils::UnivariatePoly;

use super::MetalBackend;

// MleOps implementations are required for GkrOps
impl MleOps<BaseField> for MetalBackend {
    fn fix_first_variable(mle: Mle<Self, BaseField>, assignment: SecureField) -> Mle<Self, SecureField> {
        // MLE first variable fixing uses SIMD, transmute types
        let simd_mle: Mle<SimdBackend, BaseField> = unsafe { std::mem::transmute(mle) };
        let simd_result = SimdBackend::fix_first_variable(simd_mle, assignment);
        unsafe { std::mem::transmute(simd_result) }
    }
}

impl MleOps<SecureField> for MetalBackend {
    fn fix_first_variable(mle: Mle<Self, SecureField>, assignment: SecureField) -> Mle<Self, SecureField> {
        // MLE first variable fixing uses SIMD, transmute types
        let simd_mle: Mle<SimdBackend, SecureField> = unsafe { std::mem::transmute(mle) };
        let simd_result = SimdBackend::fix_first_variable(simd_mle, assignment);
        unsafe { std::mem::transmute(simd_result) }
    }
}

impl GkrOps for MetalBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        // TODO(Phase 4): Dispatch to Metal GPU for large MLE operations
        let simd_result = SimdBackend::gen_eq_evals(y, v);
        // Transmute result from SIMD to Metal backend
        unsafe { std::mem::transmute(simd_result) }
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        // Layer transitions use SIMD, transmute types
        let simd_layer: &Layer<SimdBackend> = unsafe { &*(layer as *const _ as *const _) };
        let simd_result = SimdBackend::next_layer(simd_layer);
        unsafe { std::mem::transmute(simd_result) }
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        // Polynomial summation uses SIMD, transmute types
        let simd_h: &GkrMultivariatePolyOracle<'_, SimdBackend> =
            unsafe { &*(h as *const _ as *const _) };
        SimdBackend::sum_as_poly_in_first_variable(simd_h, claim)
    }
}
