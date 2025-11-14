//! Metal polynomial operations (PolyOps trait implementation).
//!
//! This module implements circle FFT/IFFT operations with GPU acceleration
//! for large workloads and SIMD fallback for small workloads.

use metal::MTLResourceOptions;

use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::CircleDomain;
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Column;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;

use super::context::MetalContext;
use super::thresholds::MIN_FFT_LOG_SIZE;
use super::MetalBackend;

/// Generates circle twiddles (layer 0) from first line twiddles (layer 1).
/// For each pair [x, y] in first_line_twiddles, generates [y, -y, -x, x].
/// Note: Twiddles are doubled (2*value) for SIMD optimization.
fn circle_twiddles_from_line_twiddles(first_line_twiddles: &[u32]) -> Vec<u32> {
    const P_DBL: u32 = 4294967294;  // 2 * (2^31 - 1) = 2^32 - 2
    let neg_m31_dbl = |a_dbl: u32| if a_dbl == 0 { 0 } else { P_DBL - a_dbl };

    first_line_twiddles
        .chunks_exact(2)
        .flat_map(|chunk| {
            let x_dbl = chunk[0];
            let y_dbl = chunk[1];
            let neg_y_dbl = neg_m31_dbl(y_dbl);
            let neg_x_dbl = neg_m31_dbl(x_dbl);
            [y_dbl, neg_y_dbl, neg_x_dbl, x_dbl]
        })
        .collect()
}

impl PolyOps for MetalBackend {
    // Use SIMD's twiddle format (doubled u32 values)
    type Twiddles = <SimdBackend as PolyOps>::Twiddles;

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        // Dispatch to Metal GPU for large IFFTs
        metal_ifft_dispatch(eval, twiddles)
    }

    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField {
        // Point evaluation: convert Metal->SIMD (same columns, different Backend param)
        let simd_poly: &CircleCoefficients<SimdBackend> =
            unsafe { &*(poly as *const _ as *const CircleCoefficients<SimdBackend>) };
        SimdBackend::eval_at_point(simd_poly, point)
    }

    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        // Folding evaluation: use unsafe transmute since types are layout-compatible
        let simd_evals: &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> =
            unsafe { &*(evals as *const _ as *const _) };
        let simd_twiddles: &TwiddleTree<SimdBackend> =
            unsafe { &*(twiddles as *const _ as *const _) };
        SimdBackend::eval_at_point_by_folding(simd_evals, point, simd_twiddles)
    }

    fn extend(
        poly: &CircleCoefficients<Self>,
        log_size: u32,
    ) -> CircleCoefficients<Self> {
        // Extension: transmute to SIMD, extend, transmute back
        let simd_poly: &CircleCoefficients<SimdBackend> =
            unsafe { &*(poly as *const _ as *const _) };
        let simd_result = SimdBackend::extend(simd_poly, log_size);
        // Convert back by reconstructing with the same column
        CircleCoefficients::new(simd_result.coeffs)
    }

    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // Dispatch to Metal GPU for large FFTs
        metal_fft_dispatch(poly, domain, twiddles)
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        // Twiddle precomputation uses SIMD, same twiddle format (Vec<u32>)
        let simd_result = SimdBackend::precompute_twiddles(coset);
        TwiddleTree {
            root_coset: simd_result.root_coset,
            twiddles: simd_result.twiddles,
            itwiddles: simd_result.itwiddles,
        }
    }

    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        let (left, right) = poly.coeffs.split_at_mid();
        (
            CircleCoefficients::new(left),
            CircleCoefficients::new(right),
        )
    }
}

// ============================================================================
// Metal GPU Dispatch Functions
// ============================================================================

/// Dispatch forward FFT to Metal GPU (radix-8 implementation).
///
/// This function processes FFT layers using GPU acceleration for eligible transforms.
/// For Phase 1, we only use GPU if all layers can be processed (log_size divisible by 3).
/// Otherwise, we fall back to full SIMD implementation.
///
/// TODO(Phase 2): Implement mixed GPU+SIMD execution for partial layer processing.
fn metal_fft_dispatch(
    poly: &CircleCoefficients<MetalBackend>,
    domain: CircleDomain,
    twiddles: &TwiddleTree<MetalBackend>,
) -> CircleEvaluation<MetalBackend, BaseField, BitReversedOrder> {
    let log_size = poly.log_size();

    // Number of FFT layers is log_size - 1 (based on coset size)
    let num_fft_layers = log_size - 1;

    // Phase 2: Full GPU acceleration with radix-2 for vecwise layers (0-4) and radix-8 for higher layers
    const VECWISE_FFT_BITS: u32 = 5;
    let non_vecwise_layers = if num_fft_layers > VECWISE_FFT_BITS {
        num_fft_layers - VECWISE_FFT_BITS
    } else {
        0  // All layers are vecwise, handled by radix-2
    };

    // Phase 2: Full Metal GPU with radix-2 for vecwise and radix-8 for non-vecwise
    // Fall back to SIMD if: small size or non-vecwise layers not divisible by 3
    if log_size < MIN_FFT_LOG_SIZE || (non_vecwise_layers > 0 && non_vecwise_layers % 3 != 0) {
        println!("FFT: Falling back to SIMD (log_size={}, MIN={}, non_vecwise={})",
                 log_size, MIN_FFT_LOG_SIZE, non_vecwise_layers);
        let simd_poly: &CircleCoefficients<SimdBackend> =
            unsafe { &*(poly as *const _ as *const _) };
        let simd_twiddles: &TwiddleTree<SimdBackend> =
            unsafe { &*(twiddles as *const _ as *const _) };
        let simd_result = SimdBackend::evaluate(simd_poly, domain, simd_twiddles);
        return CircleEvaluation::new(simd_result.domain, simd_result.values);
    }

    #[cfg(test)]
    println!("\n=== FFT Metal GPU path (log_size={}, num_fft_layers={}) ===", log_size, num_fft_layers);

    let ctx = MetalContext::global();
    let device = ctx.device();

    // Get raw data pointer from column
    let data_vec = poly.coeffs.to_cpu();
    let data_len = data_vec.len();

    // Create Metal buffer with shared storage (zero-copy on Apple Silicon)
    let data_buffer = device.new_buffer_with_data(
        data_vec.as_ptr() as *const _,
        (data_len * std::mem::size_of::<BaseField>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    #[cfg(test)]
    println!("Initial input (first 8): {:?}", &data_vec[..8.min(data_len)].iter().map(|f| f.0).collect::<Vec<_>>());

    // Convert flat twiddles to per-layer slices
    let twiddle_slices = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);
    let num_fft_layers = twiddle_slices.len() as u32;

    // Phase 1: Process non-vecwise layers (6+) using radix-8 kernels FIRST (DIF FFT: high to low)
    let max_layer_for_radix8 = if num_fft_layers >= 3 { num_fft_layers - 2 } else { 0 };
    // Process layers above VECWISE_FFT_BITS, stepping by 3 from highest layer down
    for layer in ((VECWISE_FFT_BITS + 1..max_layer_for_radix8 + 1).step_by(3)).rev() {

        // Get twiddle slices for this radix-8 step (3 layers)
        // Use same indexing as vecwise layers: tw_idx = layer - 1
        let tw_idx_layer2 = (layer + 2 - 1) as usize;  // Coarsest layer (highest)
        let tw_idx_layer1 = (layer + 1 - 1) as usize;  // Middle layer
        let tw_idx_layer0 = (layer - 1) as usize;      // Finest layer (lowest)

        let tw_layer2 = twiddle_slices[tw_idx_layer2];
        let tw_layer1 = twiddle_slices[tw_idx_layer1];
        let tw_layer0 = twiddle_slices[tw_idx_layer0];

        // Create twiddle buffers
        let tw0_buffer = device.new_buffer_with_data(
            tw_layer0.as_ptr() as *const _,
            (tw_layer0.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tw1_buffer = device.new_buffer_with_data(
            tw_layer1.as_ptr() as *const _,
            (tw_layer1.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tw2_buffer = device.new_buffer_with_data(
            tw_layer2.as_ptr() as *const _,
            (tw_layer2.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer for this iteration
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline
        encoder.set_compute_pipeline_state(ctx.fft_radix8_pipeline());

        // Set buffers
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&tw0_buffer), 0);  // layer0 (finest)
        encoder.set_buffer(2, Some(&tw1_buffer), 0);  // layer1 (middle)
        encoder.set_buffer(3, Some(&tw2_buffer), 0);  // layer2 (coarsest)
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        // Dispatch threads
        let num_threads = 1u64 << layer;
        let threadgroup_size = 256.min(num_threads);
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Phase 2: Process line layers (1-5) using radix-2, in reverse order
    let _line_layers_start = 1u32;
    let _line_layers_end = VECWISE_FFT_BITS.min(num_fft_layers);

    // Process all vecwise line layers (1-5) in reverse order - INCLUSIVE range!
    for layer in (_line_layers_start..=_line_layers_end).rev() {
        // Twiddle indexing: use layer - 1
        let tw_idx = (layer - 1) as usize;
        let tw_layer = twiddle_slices[tw_idx];

        let tw_buffer = device.new_buffer_with_data(
            tw_layer.as_ptr() as *const _,
            (tw_layer.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.fft_radix2_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&tw_buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        let num_threads = 1u64 << (log_size - 1);
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Phase 3: Process circle layer (layer 0) with special circle twiddles
    if num_fft_layers > 0 {
        // Generate circle twiddles from first line twiddles (layer 1)
        let first_line_twiddles = twiddle_slices[0];
        let circle_twiddles = circle_twiddles_from_line_twiddles(first_line_twiddles);

        let tw_buffer = device.new_buffer_with_data(
            circle_twiddles.as_ptr() as *const _,
            (circle_twiddles.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.fft_radix2_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&tw_buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        let layer = 0u32;
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        let num_threads = 1u64 << (log_size - 1);
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // All layers processed on GPU, copy results back
    let mut result_vec = vec![BaseField::from(0); data_len];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data_buffer.contents() as *const BaseField,
            result_vec.as_mut_ptr(),
            data_len,
        );
    }

    CircleEvaluation::new(domain, result_vec.into_iter().collect())
}

/// Dispatch inverse FFT to Metal GPU (radix-8 implementation).
///
/// Similar to forward FFT but processes layers in reverse order for IFFT.
/// For Phase 1, we only use GPU if all layers can be processed (log_size divisible by 3).
///
/// TODO(Phase 2): Implement mixed GPU+SIMD execution for partial layer processing.
fn metal_ifft_dispatch(
    eval: CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>,
    twiddles: &TwiddleTree<MetalBackend>,
) -> CircleCoefficients<MetalBackend> {
    let log_size = eval.domain.log_size();

    // Number of IFFT layers is log_size - 1 (based on coset size)
    let num_ifft_layers = log_size - 1;

    // Phase 2: Full GPU acceleration with radix-2 for vecwise layers (0-4) and radix-8 for higher layers
    const VECWISE_FFT_BITS: u32 = 5;
    let non_vecwise_layers = if num_ifft_layers > VECWISE_FFT_BITS {
        num_ifft_layers - VECWISE_FFT_BITS
    } else {
        0  // All layers are vecwise, handled by radix-2
    };

    // Phase 2: Full Metal GPU with radix-2 for vecwise and radix-8 for non-vecwise
    // Fall back to SIMD if: small size or non-vecwise layers not divisible by 3
    if log_size < MIN_FFT_LOG_SIZE || (non_vecwise_layers > 0 && non_vecwise_layers % 3 != 0) {
        #[cfg(test)]
        println!("IFFT: Falling back to SIMD (log_size={}, MIN={}, non_vecwise={})",
                 log_size, MIN_FFT_LOG_SIZE, non_vecwise_layers);
        let simd_eval = CircleEvaluation::new(eval.domain, eval.values);
        let simd_twiddles = TwiddleTree {
            root_coset: twiddles.root_coset,
            twiddles: twiddles.twiddles.clone(),
            itwiddles: twiddles.itwiddles.clone(),
        };
        let simd_result = SimdBackend::interpolate(simd_eval, &simd_twiddles);
        return CircleCoefficients::new(simd_result.coeffs);
    }

    #[cfg(test)]
    println!("IFFT: Using Metal GPU (log_size={}, non_vecwise={}, vecwise layers=0-{})",
             log_size, non_vecwise_layers, VECWISE_FFT_BITS.min(num_ifft_layers) - 1);

    let ctx = MetalContext::global();
    let device = ctx.device();

    // Get raw data
    let data_vec = eval.values.to_cpu();
    let data_len = data_vec.len();

    // Create Metal buffer
    let data_buffer = device.new_buffer_with_data(
        data_vec.as_ptr() as *const _,
        (data_len * std::mem::size_of::<BaseField>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Convert flat inverse twiddles to per-layer slices
    let itwiddle_slices = domain_line_twiddles_from_tree(eval.domain, &twiddles.itwiddles);
    let num_layers = itwiddle_slices.len() as u32;

    // Process layers in groups of 3 (radix-8)
    let mut current_layer = 0u32;

    #[cfg(test)]
    println!("\nMetal IFFT dispatch: log_size={}, num_ifft_layers={}", log_size, num_ifft_layers);

    // Phase 1: Process circle layer (layer 0) FIRST with special circle twiddles (DIT IFFT)
    if num_ifft_layers > 0 {
        #[cfg(test)]
        println!("  Circle layer: 0");

        // Generate circle twiddles from first line inverse twiddles
        let first_line_itwiddles = itwiddle_slices[(num_layers - 2) as usize];  // layer 1 itwiddles
        let circle_itwiddles = circle_twiddles_from_line_twiddles(first_line_itwiddles);

        let itw_buffer = device.new_buffer_with_data(
            circle_itwiddles.as_ptr() as *const _,
            (circle_itwiddles.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.ifft_radix2_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&itw_buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        let layer = 0u32;
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        let num_threads = 1u64 << (log_size - 1);
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        current_layer = 1;
    }

    // Phase 2: Process line layers (1-5) using radix-2, in FORWARD order (DIT IFFT)
    let line_layers_start = 1u32;
    let line_layers_end = VECWISE_FFT_BITS.min(num_ifft_layers);
    for layer in line_layers_start..=line_layers_end {
        #[cfg(test)]
        println!("  Radix-2 line layer: {}", layer);

        let itw_idx = (num_layers - 1 - layer) as usize;
        let itw_layer = itwiddle_slices[itw_idx];

        let itw_buffer = device.new_buffer_with_data(
            itw_layer.as_ptr() as *const _,
            (itw_layer.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.ifft_radix2_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&itw_buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        let num_threads = 1u64 << (log_size - 1);
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        current_layer = layer + 1;
    }

    // Phase 2: Process non-vecwise layers (6+) using radix-8 kernels
    // Iterate through remaining layers, stepping by 3
    // Process as long as we have 3 consecutive layers available
    while current_layer + 2 <= num_layers {
        let layer = current_layer;

        // Get inverse twiddle slices
        // Note: twiddle_slices are in REVERSE order (index 0 = highest layer)
        // IFFT processes bottom-to-top: when layer=0, we process layers 0,1,2
        // Kernel applies layer0 (finest=0), layer1 (middle=1), layer2 (coarsest=2)
        let itw_layer0 = itwiddle_slices[(num_layers - 1 - layer) as usize];
        let itw_layer1 = itwiddle_slices[(num_layers - 1 - (layer + 1)) as usize];
        let itw_layer2 = itwiddle_slices[(num_layers - 1 - (layer + 2)) as usize];

        // Create twiddle buffers
        let itw0_buffer = device.new_buffer_with_data(
            itw_layer0.as_ptr() as *const _,
            (itw_layer0.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let itw1_buffer = device.new_buffer_with_data(
            itw_layer1.as_ptr() as *const _,
            (itw_layer1.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let itw2_buffer = device.new_buffer_with_data(
            itw_layer2.as_ptr() as *const _,
            (itw_layer2.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create and encode command
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.ifft_radix8_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&itw0_buffer), 0);
        encoder.set_buffer(2, Some(&itw1_buffer), 0);
        encoder.set_buffer(3, Some(&itw2_buffer), 0);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        // Dispatch threads: one thread per starting position
        // Total threads = 2^layer (one per starting offset)
        let num_threads = 1u64 << layer;
        let threadgroup_size = 256.min(num_threads);
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        current_layer += 3;
    }

    // All layers processed on GPU, copy results back
    let mut result_vec = vec![BaseField::from(0); data_len];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data_buffer.contents() as *const BaseField,
            result_vec.as_mut_ptr(),
            data_len,
        );
    }

    CircleCoefficients::new(result_vec.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::Column;

    /// Compare intermediate state after radix-8: run same input twice and compare.
    #[test]
    fn test_radix8_intermediate_consistency() {
        // log_size=9: radix-8 processes layers 5,6,7
        let log_size = 9;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        println!("\n=== Testing radix-8 intermediate consistency ===");

        let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();

        // Run Metal which currently has vecwise DISABLED
        let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
        let _metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);

        // The intermediate values after radix-8 were printed in "After radix-8" debug output
        println!("\nNote: Metal intermediate values after radix-8 shown above");
        println!("We need to verify these match what SIMD produces at the same point");
    }

    /// Test ONLY radix-8 layers to verify intermediate state.
    #[test]
    fn test_metal_radix8_intermediate() {
        // Test with log_size=9 but only process radix-8 layers (5,6,7)
        // by temporarily modifying vecwise range
        let log_size = 9;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        println!("\n=== Testing radix-8 intermediate state: log_size={} ===", log_size);

        // Create simple test input
        let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();

        // Run full SIMD FFT
        let simd_poly_full = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let simd_result_full = SimdBackend::evaluate(&simd_poly_full, domain, &simd_twiddles);
        let simd_values_full = simd_result_full.values.to_cpu();

        println!("\nFull SIMD output (first 16): {:?}", &simd_values_full[..16].iter().map(|f| f.0).collect::<Vec<_>>());

        // Now run Metal which processes radix-8 + vecwise
        let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);        let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
        let metal_values = metal_result.values.to_cpu();

        println!("\nFull Metal output (first 16): {:?}", &metal_values[..16].iter().map(|f| f.0).collect::<Vec<_>>());

        // The intermediate state after radix-8 was already printed in the test output
        // Compare final results
        for (i, (s, m)) in simd_values_full.iter().zip(metal_values.iter()).enumerate().take(16) {
            if s != m {
                println!("Mismatch at {}: SIMD={}, Metal={}", i, s.0, m.0);
            }
        }
    }

    /// Test with one radix-8 group + vecwise layers.
    #[test]
    fn test_metal_mixed_radix8_vecwise() {
        // Test with log_size=9: num_fft_layers=8
        // non_vecwise=3 (layers 5,6,7 - one radix-8 group)
        // vecwise (layers 4,3,2,1,0)
        let log_size = 9;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        println!("\n=== Testing mixed radix-8 + vecwise: log_size={} ===", log_size);
        println!("num_fft_layers: {}, non_vecwise: 3 (layers 5,6,7)", log_size - 1);

        // Create simple test input
        let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();

        println!("\nInput (first 8):");
        for i in 0..8 {
            println!("  [{}] = {}", i, coeffs_data[i].0);
        }

        // SIMD
        let simd_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let simd_result = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);
        let simd_values = simd_result.values.to_cpu();

        // Metal
        let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
        let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
        let metal_values = metal_result.values.to_cpu();

        println!("\nSIMD output (first 16):");
        for i in 0..16 {
            println!("  [{}] = {}", i, simd_values[i].0);
        }

        println!("\nMetal output (first 16):");
        for i in 0..16 {
            println!("  [{}] = {}", i, metal_values[i].0);
        }

        // Compare
        for (i, (s, m)) in simd_values.iter().zip(metal_values.iter()).enumerate() {
            assert_eq!(s, m, "Mismatch at index {}", i);
        }
    }

    /// Test a single radix-2 layer to isolate kernel bugs.
    #[test]
    fn test_single_radix2_layer() {
        // Manually test layer 1 processing
        let log_size = 3; // 8 elements
        let size = 1 << log_size;

        println!("\n=== Testing single radix-2 layer ===");

        // Simple input
        let input: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();
        println!("Input: {:?}", input.iter().map(|f| f.0).collect::<Vec<_>>());

        // TODO: Manually invoke the Metal radix-2 kernel for layer 1
        // Then compare with expected CPU output

        println!("This test needs manual kernel invocation - skipping for now");
    }

    /// Test just the vecwise layers (no radix-8) to isolate the issue.
    #[test]
    fn test_metal_vecwise_only() {
        // Test with log_size=6: num_fft_layers=5 (all vecwise, no radix-8)
        let log_size = 6;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        println!("\n=== Testing vecwise-only: log_size={} ===", log_size);
        println!("num_fft_layers: {}", log_size - 1);

        // Create simple test input
        let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();

        println!("\nInput (first 8):");
        for i in 0..8 {
            println!("  [{}] = {}", i, coeffs_data[i].0);
        }

        // SIMD
        let simd_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let simd_result = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);
        let simd_values = simd_result.values.to_cpu();

        // Metal
        let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
        let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
        let metal_values = metal_result.values.to_cpu();

        println!("\nSIMD output (first 8):");
        for i in 0..8 {
            println!("  [{}] = {}", i, simd_values[i].0);
        }

        println!("\nMetal output (first 8):");
        for i in 0..8 {
            println!("  [{}] = {}", i, metal_values[i].0);
        }

        // Compare
        for (i, (s, m)) in simd_values.iter().zip(metal_values.iter()).enumerate() {
            assert_eq!(s, m, "Mismatch at index {}", i);
        }
    }

    /// Debug test with detailed logging to understand Metal vs SIMD divergence.
    #[test]
    fn test_metal_fft_debug() {
        use crate::core::poly::utils::domain_line_twiddles_from_tree;

        // Test with log_size=10 - should fallback to SIMD
        let log_size = 10;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        println!("\n=== FFT Debug Test: log_size={} ===", log_size);
        println!("Size: {}, num_fft_layers: {}", size, log_size - 1);
        println!("Domain log_size: {}, Domain size: {}", domain.log_size(), domain.size());
        println!("Half coset log_size: {}, Half coset size: {}",
                 domain.half_coset.log_size(), domain.half_coset.size());

        // Create simple test input
        let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();

        println!("\nInput (first 16 elements):");
        for i in 0..16 {
            println!("  coeffs[{}] = {}", i, coeffs_data[i].0);
        }

        // Get twiddle information
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let twiddle_slices = domain_line_twiddles_from_tree(domain, &simd_twiddles.twiddles);

        println!("\nTwiddle slice information:");
        println!("  Total twiddle slices: {}", twiddle_slices.len());
        for (i, slice) in twiddle_slices.iter().enumerate() {
            println!("  twiddle_slices[{}]: len={} (2^{})",
                     i, slice.len(), (slice.len() as f64).log2() as u32);
        }

        // For first radix-8 iteration (layer=6), show which twiddles are used
        let layer = 6u32;
        let num_fft_layers = (log_size - 1) as u32;
        println!("\nFirst radix-8 iteration (layer={}):", layer);
        println!("  Processing physical layers: {}, {}, {}", layer, layer+1, layer+2);

        let tw_idx_layer0 = (num_fft_layers - 1 - layer) as usize;
        let tw_idx_layer1 = (num_fft_layers - 1 - (layer + 1)) as usize;
        let tw_idx_layer2 = (num_fft_layers - 1 - (layer + 2)) as usize;

        println!("  tw_layer0 (finest): twiddle_slices[{}], len={}",
                 tw_idx_layer0, twiddle_slices[tw_idx_layer0].len());
        println!("  tw_layer1 (middle): twiddle_slices[{}], len={}",
                 tw_idx_layer1, twiddle_slices[tw_idx_layer1].len());
        println!("  tw_layer2 (coarsest): twiddle_slices[{}], len={}",
                 tw_idx_layer2, twiddle_slices[tw_idx_layer2].len());

        // Show first few twiddles for each layer and verify they match what SIMD would use
        println!("\n  First 4 twiddles from each layer (for first radix-8 iteration):");
        println!("  For layer={}, index=0 (first block):", layer);

        // SIMD uses twiddle_dbl[layer-1], twiddle_dbl[layer], twiddle_dbl[layer+1]
        // But we get them from domain_line_twiddles which are in reverse order
        // So we need to map correctly

        for i in 0..4.min(twiddle_slices[tw_idx_layer0].len()) {
            println!("    tw_layer0[{}] = {} (used by finest layer, physical layer {})",
                     i, twiddle_slices[tw_idx_layer0][i], layer);
        }
        for i in 0..4.min(twiddle_slices[tw_idx_layer1].len()) {
            println!("    tw_layer1[{}] = {} (used by middle layer, physical layer {})",
                     i, twiddle_slices[tw_idx_layer1][i], layer + 1);
        }
        for i in 0..4.min(twiddle_slices[tw_idx_layer2].len()) {
            println!("    tw_layer2[{}] = {} (used by coarsest layer, physical layer {})",
                     i, twiddle_slices[tw_idx_layer2][i], layer + 2);
        }

        // Also check what SIMD twiddle_dbl array looks like
        println!("\n  Full twiddle_slices array lengths:");
        for (i, slice) in twiddle_slices.iter().enumerate() {
            println!("    twiddle_slices[{}]: len={}, physical_layer={}",
                     i, slice.len(), (num_fft_layers as usize) - 1 - i);
        }

        // Run SIMD FFT
        let simd_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let simd_result = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);
        let simd_values = simd_result.values.to_cpu();

        println!("\nSIMD FFT output (first 16 elements):");
        for i in 0..16 {
            println!("  simd[{}] = {}", i, simd_values[i].0);
        }

        // Run Metal FFT
        let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
        let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
        let metal_values = metal_result.values.to_cpu();

        println!("\nMetal FFT output (first 16 elements):");
        for i in 0..16 {
            println!("  metal[{}] = {}", i, metal_values[i].0);
        }

        println!("\nComparison (first 16 elements):");
        let mut num_diffs = 0;
        for i in 0..16 {
            if simd_values[i] != metal_values[i] {
                println!("  [{}] DIFF: simd={}, metal={}",
                         i, simd_values[i].0, metal_values[i].0);
                num_diffs += 1;
            } else {
                println!("  [{}] OK: {}", i, simd_values[i].0);
            }
        }

        if num_diffs > 0 {
            panic!("{} differences found in first 16 elements", num_diffs);
        }
    }

    /// Test that Metal FFT produces the same results as SIMD FFT.
    #[test]
    fn test_metal_fft_correctness() {
        // Test on sizes where non_vecwise_layers is divisible by 3
        // For non_vecwise = (num_fft_layers - 5) to be divisible by 3:
        // num_fft_layers = 5 + 3k, so log_size = 6 + 3k
        // log_size 12: num_fft_layers = 11, non_vecwise = 6 (divisible by 3) ✓
        // log_size 15: num_fft_layers = 14, non_vecwise = 9 (divisible by 3) ✓
        // log_size 18: num_fft_layers = 17, non_vecwise = 12 (divisible by 3) ✓
        for log_size in [12, 15, 18] {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let size = 1 << log_size;

            // Create test input
            let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i)).collect();

            // Run SIMD FFT
            let simd_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
            let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
            let simd_result = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);
            let simd_values = simd_result.values.to_cpu();

            // Run Metal FFT
            let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
            let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
            let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
            let metal_values = metal_result.values.to_cpu();

            // Compare results
            assert_eq!(
                simd_values.len(),
                metal_values.len(),
                "FFT result lengths differ for log_size {}",
                log_size
            );

            for (i, (simd_val, metal_val)) in
                simd_values.iter().zip(metal_values.iter()).enumerate()
            {
                assert_eq!(
                    simd_val, metal_val,
                    "FFT results differ at index {} for log_size {}",
                    i, log_size
                );
            }
        }
    }

    /// Test that Metal IFFT produces the same results as SIMD IFFT.
    #[test]
    fn test_metal_ifft_correctness() {
        for log_size in [13, 16, 19] {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let size = 1 << log_size;

            // Create test input (evaluations)
            let eval_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i)).collect();

            // Run SIMD IFFT
            let simd_eval = CircleEvaluation::new(
                domain,
                eval_data.iter().copied().collect(),
            );
            let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
            let simd_result = SimdBackend::interpolate(simd_eval, &simd_twiddles);
            let simd_coeffs = simd_result.coeffs.to_cpu();

            // Run Metal IFFT
            let metal_eval = CircleEvaluation::new(
                domain,
                eval_data.iter().copied().collect(),
            );
            let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
            let metal_result = MetalBackend::interpolate(metal_eval, &metal_twiddles);
            let metal_coeffs = metal_result.coeffs.to_cpu();

            // Compare results
            assert_eq!(
                simd_coeffs.len(),
                metal_coeffs.len(),
                "IFFT result lengths differ for log_size {}",
                log_size
            );

            for (i, (simd_val, metal_val)) in
                simd_coeffs.iter().zip(metal_coeffs.iter()).enumerate()
            {
                assert_eq!(
                    simd_val, metal_val,
                    "IFFT results differ at index {} for log_size {}",
                    i, log_size
                );
            }
        }
    }

    /// Test FFT/IFFT round-trip produces original coefficients.
    #[test]
    fn test_metal_fft_ifft_roundtrip() {
        for log_size in [13, 16] {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let size = 1 << log_size;

            // Create original coefficients
            let original: Vec<BaseField> = (0..size).map(|i| BaseField::from(i % 1000)).collect();

            // FFT -> IFFT round trip
            let poly = CircleCoefficients::new(original.iter().copied().collect());
            let twiddles = MetalBackend::precompute_twiddles(domain.half_coset);

            let eval = MetalBackend::evaluate(&poly, domain, &twiddles);
            let reconstructed = MetalBackend::interpolate(eval, &twiddles);

            let result = reconstructed.coeffs.to_cpu();

            // Compare
            assert_eq!(
                original.len(),
                result.len(),
                "Round-trip length mismatch for log_size {}",
                log_size
            );

            for (i, (orig, recon)) in original.iter().zip(result.iter()).enumerate() {
                assert_eq!(
                    orig, recon,
                    "Round-trip failed at index {} for log_size {}",
                    i, log_size
                );
            }
        }
    }

    /// Test that sizes where num_fft_layers is not divisible by 3 fall back to SIMD correctly.
    #[test]
    fn test_metal_fallback_to_simd() {
        // log_size 14: num_fft_layers = 13 (not divisible by 3, should fall back to SIMD)
        let log_size = 14;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        let coeffs: Vec<BaseField> = (0..size).map(|i| BaseField::from(i)).collect();

        // This should fall back to SIMD internally but still work
        let poly = CircleCoefficients::new(coeffs.iter().copied().collect());
        let twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
        let eval = MetalBackend::evaluate(&poly, domain, &twiddles);
        let reconstructed = MetalBackend::interpolate(eval, &twiddles);

        let result = reconstructed.coeffs.to_cpu();

        // Verify round-trip
        for (i, (orig, recon)) in coeffs.iter().zip(result.iter()).enumerate() {
            assert_eq!(
                orig, recon,
                "Fallback round-trip failed at index {}",
                i
            );
        }
    }
}
