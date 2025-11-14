//! Metal proof-of-work grinding operations (GrindOps trait implementation).
//!
//! This module implements PoW grinding, delegating to SIMD for now.
//! GPU acceleration for grinding could be added in future phases.

use crate::core::channel::Blake2sChannelGeneric;
use crate::core::proof_of_work::GrindOps;
use crate::prover::backend::simd::SimdBackend;

use super::MetalBackend;

impl<const IS_M31_OUTPUT: bool> GrindOps<Blake2sChannelGeneric<IS_M31_OUTPUT>> for MetalBackend {
    fn grind(channel: &Blake2sChannelGeneric<IS_M31_OUTPUT>, pow_bits: u32) -> u64 {
        // TODO(Phase 5): Consider GPU acceleration for grinding
        // For now, use SIMD which has good parallel implementation
        SimdBackend::grind(channel, pow_bits)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub mod poseidon252 {
    use crate::core::channel::Poseidon252Channel;
    use crate::core::proof_of_work::GrindOps;
    use crate::prover::backend::simd::SimdBackend;

    use super::MetalBackend;

    impl GrindOps<Poseidon252Channel> for MetalBackend {
        fn grind(channel: &Poseidon252Channel, pow_bits: u32) -> u64 {
            // Poseidon252 grinding uses SIMD
            SimdBackend::grind(channel, pow_bits)
        }
    }
}
