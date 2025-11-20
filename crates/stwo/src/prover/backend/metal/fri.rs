//! Metal FRI operations (FriOps trait implementation).
//!
//! This module implements FRI folding operations with GPU acceleration for large workloads
//! and SIMD fallback for small workloads.

use metal::MTLResourceOptions;

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

impl FriOps for MetalBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let log_size = eval.len().ilog2();
        let _timer = crate::metal_profile_fn!("fri_fold_line", "GPU", log_size = log_size);

        // Fall back to SIMD for small sizes
        if log_size < MIN_FRI_LOG_SIZE {
            use crate::prover::backend::Column;
            use crate::prover::backend::simd::column::BaseColumn;
            use crate::prover::secure_column::SecureColumnByCoords;

            // Convert Metal eval to SIMD
            let simd_columns = eval.values.columns.clone().map(|col| {
                let cpu_vals = col.to_cpu();
                let simd_col: BaseColumn = cpu_vals.into_iter().collect();
                simd_col
            });
            let simd_values = SecureColumnByCoords { columns: simd_columns };
            let simd_eval = LineEvaluation::new(eval.domain(), simd_values);

            let simd_twiddles: &TwiddleTree<SimdBackend> =
                unsafe { &*(twiddles as *const _ as *const _) };
            let simd_result = SimdBackend::fold_line(&simd_eval, alpha, simd_twiddles);

            // Convert result back to Metal
            let domain = simd_result.domain();
            let metal_values = SecureColumnByCoords::from_simd(simd_result.values);
            return LineEvaluation::new(domain, metal_values);
        }

        let ctx = MetalContext::global();
        let device = ctx.device();

        let domain = eval.domain();
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        // Convert input to interleaved QM31 layout - zero-copy via shared buffer
        let input_buffer = eval.values.as_qm31_interleaved_u32_buffer();
        let output_len = 1 << (log_size - 1);

        // Create output buffer using buffer pool
        let output_size = (output_len * 4 * std::mem::size_of::<u32>()) as u64;
        let output_pooled = ctx.checkout_shared_buffer(output_size);
        let output_buffer = output_pooled.buffer();

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

        // Convert output buffer directly to SecureColumnByCoords - zero-copy
        let folded_values = unsafe {
            SecureColumnByCoords::from_qm31_interleaved_buffer(&output_buffer, output_len)
        };

        // Keep pooled buffer alive until after GPU completes and data is read
        drop(output_pooled);

        LineEvaluation::new(domain.double(), folded_values)
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        let log_size = src.len().ilog2();
        let _timer = crate::metal_profile_fn!("fri_fold_circle", "GPU", log_size = log_size);

        // Fall back to SIMD for small sizes
        if log_size < MIN_FRI_LOG_SIZE {
            use crate::prover::backend::Column;
            use crate::prover::backend::simd::column::BaseColumn;

            // Convert dst and src from Metal to SIMD
            let dst_domain = dst.domain();
            let simd_dst_columns = dst.values.columns.clone().map(|col| {
                let cpu_vals = col.to_cpu();
                let simd_col: BaseColumn = cpu_vals.into_iter().collect();
                simd_col
            });
            let simd_dst_values = SecureColumnByCoords { columns: simd_dst_columns };
            let mut simd_dst = LineEvaluation::new(dst_domain, simd_dst_values);

            let simd_src_columns = src.values.columns.clone().map(|col| {
                let cpu_vals = col.to_cpu();
                let simd_col: BaseColumn = cpu_vals.into_iter().collect();
                simd_col
            });
            let simd_src_values = SecureColumnByCoords { columns: simd_src_columns };
            let simd_src = SecureEvaluation::new(src.domain, simd_src_values);

            let simd_twiddles: &TwiddleTree<SimdBackend> =
                unsafe { &*(twiddles as *const _ as *const _) };

            SimdBackend::fold_circle_into_line(&mut simd_dst, &simd_src, alpha, simd_twiddles);

            // Write result back to dst
            dst.values = SecureColumnByCoords::from_simd(simd_dst.values);
            return;
        }

        // Metal GPU path
        let ctx = MetalContext::global();
        let device = ctx.device();

        let domain = src.domain;
        let alpha_sq = alpha * alpha;
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        // Convert source and dst to interleaved QM31 layout - zero-copy via shared buffer
        let src_buffer = src.values.as_qm31_interleaved_u32_buffer();
        let dst_buffer = dst.values.as_qm31_interleaved_u32_buffer();
        let output_len = 1 << (log_size - 1);

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

        // Convert output buffer directly to SecureColumnByCoords - zero-copy
        dst.values = unsafe {
            SecureColumnByCoords::from_qm31_interleaved_buffer(&dst_buffer, output_len)
        };
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        use crate::prover::backend::Column;
        use crate::prover::backend::simd::column::BaseColumn;
        use crate::prover::secure_column::SecureColumnByCoords;

        // Convert Metal eval to SIMD
        let simd_columns = eval.values.columns.clone().map(|col| {
            let cpu_vals = col.to_cpu();
            let simd_col: BaseColumn = cpu_vals.into_iter().collect();
            simd_col
        });
        let simd_values = SecureColumnByCoords { columns: simd_columns };
        let simd_eval = SecureEvaluation::new(eval.domain, simd_values);

        let (simd_result, field) = SimdBackend::decompose(&simd_eval);

        // Convert result back to Metal
        let metal_values = SecureColumnByCoords::from_simd(simd_result.values);
        let metal_result = SecureEvaluation::new(simd_result.domain, metal_values);
        (metal_result, field)
    }
}
