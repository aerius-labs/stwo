//! Metal FRI operations (FriOps trait implementation).
//!
//! This module implements FRI folding operations with GPU acceleration for large workloads
//! and SIMD fallback for small workloads.

use metal::MTLResourceOptions;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::fri::FriOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

use super::context::MetalContext;
use super::thresholds::MIN_FRI_LOG_SIZE;
use super::MetalBackend;

/// Convert SecureColumnByCoords to interleaved QM31 layout for GPU
/// Layout: [QM31, QM31, ...] where each QM31 = {CM31 c0, CM31 c1}
/// and CM31 = {u32 a, u32 b}
fn secure_column_to_qm31_buffer(column: &SecureColumnByCoords<MetalBackend>) -> Vec<u32> {
    let len = column.len();
    let mut buffer = Vec::with_capacity(len * 4); // 4 u32s per SecureField

    for i in 0..len {
        let value = column.at(i);
        let [a, b, c, d] = value.to_m31_array();
        buffer.push(a.0);
        buffer.push(b.0);
        buffer.push(c.0);
        buffer.push(d.0);
    }

    buffer
}

/// Convert interleaved QM31 buffer back to SecureColumnByCoords
fn qm31_buffer_to_secure_column(buffer: &[u32], len: usize) -> SecureColumnByCoords<MetalBackend> {
    let mut column = SecureColumnByCoords::zeros(len);

    for i in 0..len {
        let a = BaseField::from(buffer[i * 4]);
        let b = BaseField::from(buffer[i * 4 + 1]);
        let c = BaseField::from(buffer[i * 4 + 2]);
        let d = BaseField::from(buffer[i * 4 + 3]);
        let value = SecureField::from_m31(a, b, c, d);
        column.set(i, value);
    }

    column
}

impl FriOps for MetalBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let log_size = eval.len().ilog2();

        // Fall back to SIMD for small sizes
        if log_size < MIN_FRI_LOG_SIZE {
            let simd_eval: &LineEvaluation<SimdBackend> =
                unsafe { &*(eval as *const _ as *const _) };
            let simd_twiddles: &TwiddleTree<SimdBackend> =
                unsafe { &*(twiddles as *const _ as *const _) };
            let simd_result = SimdBackend::fold_line(simd_eval, alpha, simd_twiddles);
            let metal_result: LineEvaluation<MetalBackend> =
                unsafe { std::mem::transmute(simd_result) };
            return metal_result;
        }

        let ctx = MetalContext::global();
        let device = ctx.device();

        let domain = eval.domain();
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        // Convert input to interleaved QM31 layout
        let input_data = secure_column_to_qm31_buffer(&eval.values);
        let output_len = 1 << (log_size - 1);

        // Create Metal buffers
        let input_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const _,
            (input_data.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_data_zeros = vec![0u32; output_len * 4];
        let output_buffer = device.new_buffer_with_data(
            output_data_zeros.as_ptr() as *const _,
            (output_len * 4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Twiddles are M31 (u32) in doubled format - use cache
        let twiddle_buffer = ctx.get_or_create_twiddle_buffer(itwiddles);

        // Create alpha buffer (QM31)
        let alpha_data = alpha.to_m31_array().map(|m| m.0);
        let alpha_buffer = device.new_buffer_with_data(
            alpha_data.as_ptr() as *const _,
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch kernel
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.fri_fold_line_pipeline());
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&twiddle_buffer), 0);
        encoder.set_buffer(3, Some(&alpha_buffer), 0);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);

        let num_threads = output_len as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back results
        let output_data = unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const u32,
                output_len * 4,
            )
        };

        let folded_values = qm31_buffer_to_secure_column(output_data, output_len);

        LineEvaluation::new(domain.double(), folded_values)
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        let log_size = src.len().ilog2();

        // Fall back to SIMD for small sizes
        if log_size < MIN_FRI_LOG_SIZE {
            let simd_dst: &mut LineEvaluation<SimdBackend> =
                unsafe { &mut *(dst as *mut _ as *mut _) };
            let simd_src: &SecureEvaluation<SimdBackend, BitReversedOrder> =
                unsafe { &*(src as *const _ as *const _) };
            let simd_twiddles: &TwiddleTree<SimdBackend> =
                unsafe { &*(twiddles as *const _ as *const _) };
            SimdBackend::fold_circle_into_line(simd_dst, simd_src, alpha, simd_twiddles);
            return;
        }

        // Metal GPU path
        let ctx = MetalContext::global();
        let device = ctx.device();

        let domain = src.domain;
        let alpha_sq = alpha * alpha;
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        // Convert source to interleaved QM31 layout
        let src_data = secure_column_to_qm31_buffer(&src.values);
        let dst_data = secure_column_to_qm31_buffer(&dst.values);
        let output_len = 1 << (log_size - 1);

        // Create Metal buffers
        let src_buffer = device.new_buffer_with_data(
            src_data.as_ptr() as *const _,
            (src_data.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let dst_buffer = device.new_buffer_with_data(
            dst_data.as_ptr() as *const _,
            (dst_data.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Twiddles are M31 (u32) in doubled format - use cache
        let twiddle_buffer = ctx.get_or_create_twiddle_buffer(itwiddles);

        // Create alpha and alpha_sq buffers (QM31)
        let alpha_data = alpha.to_m31_array().map(|m| m.0);
        let alpha_buffer = device.new_buffer_with_data(
            alpha_data.as_ptr() as *const _,
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let alpha_sq_data = alpha_sq.to_m31_array().map(|m| m.0);
        let alpha_sq_buffer = device.new_buffer_with_data(
            alpha_sq_data.as_ptr() as *const _,
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch kernel
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.fri_fold_circle_pipeline());
        encoder.set_buffer(0, Some(&src_buffer), 0);
        encoder.set_buffer(1, Some(&dst_buffer), 0);
        encoder.set_buffer(2, Some(&twiddle_buffer), 0);
        encoder.set_buffer(3, Some(&alpha_buffer), 0);
        encoder.set_buffer(4, Some(&alpha_sq_buffer), 0);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);

        let num_threads = output_len as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back results into dst
        let output_data = unsafe {
            std::slice::from_raw_parts(
                dst_buffer.contents() as *const u32,
                output_len * 4,
            )
        };

        dst.values = qm31_buffer_to_secure_column(output_data, output_len);
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
