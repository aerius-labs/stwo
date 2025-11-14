//! Integration test for Metal backend.
//!
//! This test verifies that the Metal backend can be instantiated and used
//! as a drop-in replacement for other backends.

#![cfg(all(target_os = "macos", feature = "metal_prover"))]

use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::metal::MetalBackend;
use stwo::prover::backend::{Column, ColumnOps};

#[test]
fn test_metal_backend_is_usable() {
    // Phase 0: MetalBackend is a type alias to SimdBackend
    // This test verifies the backend can be used normally

    // Create a simple column through the backend's ColumnOps trait
    type MetalColumn = <MetalBackend as ColumnOps<BaseField>>::Column;

    let values: Vec<BaseField> = (0..16)
        .map(|i| BaseField::from(i))
        .collect();

    let col: MetalColumn = values.into_iter().collect();

    assert_eq!(col.len(), 16);
    assert_eq!(col.at(0), BaseField::from(0));
    assert_eq!(col.at(15), BaseField::from(15));
}

#[test]
fn test_metal_context_available() {
    use stwo::prover::backend::metal::MetalContext;

    // Check if Metal is available on this system
    let available = MetalContext::is_available();

    // On macOS with Metal support, this should be true
    // The test doesn't assert since CI might not have Metal
    println!("Metal available: {}", available);

    if available {
        // If Metal is available, we can create a context
        let ctx = MetalContext::global();
        println!("Metal device: {}", ctx.device().name());
    }
}

#[test]
fn test_metal_backend_column_operations() {
    type MetalColumn = <MetalBackend as ColumnOps<BaseField>>::Column;

    // Test from_iter (SIMD backend's primary column creation method)
    let values: Vec<BaseField> = (0..16).map(BaseField::from).collect();
    let col: MetalColumn = values.into_iter().collect();

    assert_eq!(col.len(), 16);

    // Verify we can read values back
    let cpu_values = col.to_cpu();
    assert_eq!(cpu_values.len(), 16);
    assert_eq!(cpu_values[0], BaseField::from(0));
    assert_eq!(cpu_values[15], BaseField::from(15));
}

#[test]
fn test_metal_infrastructure_exports() {
    // Verify all Metal infrastructure is exported and accessible
    use stwo::prover::backend::metal::{
        MetalBackend, MetalBaseColumn, MetalContext, MetalContextHandle, MetalSecureColumn,
    };

    // These types should all be accessible
    let _backend: Option<MetalBackend> = None;
    let _base_col: Option<MetalBaseColumn> = None;
    let _secure_col: Option<MetalSecureColumn> = None;
    let _ctx: Option<MetalContext> = None;
    let _handle: Option<MetalContextHandle> = None;

    // Test passed if compilation succeeds
}
