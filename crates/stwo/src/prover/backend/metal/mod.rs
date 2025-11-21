//! Metal GPU-accelerated backend for Apple Silicon.
//!
//! This backend leverages Metal compute shaders to accelerate:
//! - FFT/IFFT operations (circle polynomials)
//! - FRI folding (line and circle)
//! - Quotient accumulation
//! - Merkle tree construction
//! - Lookup operations (MLE/GKR)
//! - Proof-of-work grinding
//!
//! # Memory Model
//!
//! Uses `MTLStorageModeShared` for unified memory access between CPU and GPU,
//! eliminating explicit copies on Apple Silicon.
//!
//! # Pipelining Strategy
//!
//! - Command buffers can be enqueued while previous ones execute
//! - Threadgroup (shared) memory used for FFT tiles and reductions
//! - Falls back to SIMD/CPU for small workloads (below threshold)
//!
//! # Implementation Status (Phase 0 - Scaffolding)
//!
//! Currently, this backend uses SIMD implementations internally. Metal GPU kernels and
//! custom column types will be implemented incrementally in later phases. The Metal
//! infrastructure (context, shaders, custom columns) is ready but not yet integrated
//! into the main execution path.

#[cfg(target_os = "macos")]
mod context;
#[cfg(target_os = "macos")]
mod column;
#[cfg(target_os = "macos")]
mod shaders;
#[cfg(target_os = "macos")]
mod twiddle_manager;
#[cfg(target_os = "macos")]
mod buffer_pool;
#[cfg(target_os = "macos")]
pub mod profiling;

// Export Metal context (actively used)
#[cfg(target_os = "macos")]
pub use context::{MetalContext, MetalContextHandle};

// Export Metal column types (now actively used)
#[cfg(target_os = "macos")]
pub use column::{GpuSlice, MetalBaseColumn, MetalSecureColumn};

#[cfg(target_os = "macos")]
use serde::{Deserialize, Serialize};
#[cfg(target_os = "macos")]
use crate::core::fields::m31::BaseField;
#[cfg(target_os = "macos")]
use crate::core::fields::qm31::SecureField;
#[cfg(target_os = "macos")]
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleChannel, Blake2sMerkleChannel};
#[cfg(target_os = "macos")]
use crate::prover::backend::{Backend, BackendForChannel, ColumnOps};

/// Size thresholds for GPU vs CPU dispatch.
///
/// These thresholds are tuned based on benchmarks to ensure GPU is only used
/// when it provides a performance benefit over SIMD. Below these thresholds,
/// GPU dispatch overhead dominates and SIMD is faster.
#[cfg(target_os = "macos")]
pub mod thresholds {
    /// Minimum log size for GPU FFT.
    /// Lowered to 12 for batched operations and improved kernels.
    pub const MIN_FFT_LOG_SIZE: u32 = 12;

    /// Minimum log size for GPU FRI folding.
    /// Lowered to 10 for better GPU utilization in batched contexts.
    pub const MIN_FRI_LOG_SIZE: u32 = 10;

    /// Minimum log size for GPU Merkle operations.
    pub const MIN_MERKLE_LOG_SIZE: u32 = 10;

    /// Minimum log size for GPU quotient accumulation.
    /// Lowered to 10 for better GPU coverage.
    pub const MIN_QUOTIENT_LOG_SIZE: u32 = 10;

    /// Minimum log size for GPU MLE operations.
    /// Lowered to 10 for better GPU coverage.
    pub const MIN_MLE_LOG_SIZE: u32 = 10;
}

// Trait implementations
#[cfg(target_os = "macos")]
mod poly;
#[cfg(target_os = "macos")]
mod fri;
#[cfg(target_os = "macos")]
mod quotients;
#[cfg(target_os = "macos")]
mod accumulation;
#[cfg(target_os = "macos")]
mod gkr;
#[cfg(target_os = "macos")]
mod merkle;
#[cfg(target_os = "macos")]
mod grind;

/// Metal GPU-accelerated backend.
///
/// This backend uses Metal GPU compute shaders for large workloads and falls back
/// to SIMD/CPU for small workloads below size thresholds.
#[cfg(target_os = "macos")]
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct MetalBackend;

#[cfg(target_os = "macos")]
impl Backend for MetalBackend {}

#[cfg(target_os = "macos")]
impl BackendForChannel<Blake2sMerkleChannel> for MetalBackend {}

#[cfg(target_os = "macos")]
impl BackendForChannel<Blake2sM31MerkleChannel> for MetalBackend {}

// ============================================================================
// PHASE 1 ARCHITECTURE DECISION: MetalBackend = "SIMD + GPU Helpers"
// ============================================================================
//
// For Phase 1, MetalBackend uses SIMD's column types (BaseColumn/SecureColumn)
// and delegates most operations to SimdBackend, with GPU acceleration for:
// - FFT/IFFT (metal/poly.rs)
// - FRI folding (metal/fri.rs)
// - Quotient accumulation (metal/quotients.rs)
// - Merkle tree hashing (metal/merkle.rs)
// - PoW grinding (metal/grind.rs)
// - GKR MLE operations (metal/gkr.rs)
//
// This design allows Metal GPU kernels to work on data copied from SIMD columns,
// avoiding the complexity of managing GPU-backed column types in Phase 1.
//
// MetalBaseColumn/MetalSecureColumn exist in column.rs but are NOT used in the
// actual proving path - they're experimental types for future Phase 2 optimization.
//
// IMPORTANT: All `unsafe transmute` between SimdBackend and MetalBackend types
// are SAFE because both use identical column representations (BaseColumn/SecureColumn).
// However, this invariant MUST be maintained - if column types diverge, transmutes
// become UB.
//
// Future Phase 2: Switch to GPU-backed columns for zero-copy Metal operations.
// ============================================================================

#[cfg(target_os = "macos")]
impl ColumnOps<BaseField> for MetalBackend {
    type Column = MetalBaseColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        use crate::core::utils::bit_reverse;
        let slice = column.as_mut_slice();
        bit_reverse(slice);
    }
}

#[cfg(target_os = "macos")]
impl ColumnOps<SecureField> for MetalBackend {
    type Column = MetalSecureColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        use crate::core::utils::bit_reverse;
        let slice = column.as_mut_slice();
        bit_reverse(slice);
    }
}

// Placeholder implementations when not on macOS
#[cfg(not(target_os = "macos"))]
pub struct MetalBackend;

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub fn is_available() -> bool {
        false
    }
}
