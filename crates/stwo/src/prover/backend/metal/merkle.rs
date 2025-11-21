//! Metal Merkle operations (MerkleOps trait implementation).
//!
//! This module implements Merkle tree construction with GPU acceleration for large workloads
//! and SIMD fallback for small workloads.

use metal::MTLResourceOptions;

use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleHasher, Blake2sMerkleHasher};
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::{Col, ColumnOps};
use crate::prover::vcs::ops::MerkleOps;

use super::context::MetalContext;
use super::thresholds::MIN_MERKLE_LOG_SIZE;
use super::MetalBackend;

/// Build complete Merkle tree in a single GPU submission.
/// This is more efficient than layer-by-layer construction as it:
/// 1. Pre-allocates all buffers upfront
/// 2. Submits all layers in a single command buffer
/// 3. Lets Metal driver schedule layer dependencies automatically
#[allow(dead_code)]
fn commit_tree_batched<H: Into<bool>>(
    ctx: &MetalContext,
    initial_layer: Vec<Blake2sHash>,
    num_layers: u32,
    is_m31: H,
) -> Vec<Vec<Blake2sHash>> {
    if num_layers == 0 || initial_layer.is_empty() {
        return vec![];
    }

    let is_m31_output = is_m31.into();
    let device = ctx.device();

    // Pre-allocate all buffers for the entire tree
    let mut layer_buffers = Vec::new();
    let mut current_size = initial_layer.len();

    // First layer is input
    let mut initial_data = Vec::with_capacity(initial_layer.len() * 8);
    for hash in &initial_layer {
        let hash_u32s: &[u32; 8] = bytemuck::cast_ref(&hash.0);
        initial_data.extend_from_slice(hash_u32s);
    }

    let initial_buffer = device.new_buffer_with_data(
        initial_data.as_ptr() as *const _,
        (initial_data.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    layer_buffers.push((current_size, initial_buffer));

    // Pre-allocate buffers for all subsequent layers
    for _ in 0..num_layers {
        current_size /= 2;
        let buffer = device.new_buffer(
            (current_size * 8 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        layer_buffers.push((current_size, buffer));
    }

    // Create a single command buffer for all layers
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Process each layer
    for i in 0..num_layers as usize {
        let (_, input_buffer) = &layer_buffers[i];
        let (num_parents, output_buffer) = &layer_buffers[i + 1];
        let size_param = *num_parents as u32;

        encoder.set_compute_pipeline_state(ctx.merkle_pipeline());
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<bool>() as u64,
            &is_m31_output as *const bool as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &size_param as *const u32 as *const _,
        );

        let num_threads = *num_parents as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: threadgroups,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            },
        );
    }

    // Submit all operations at once
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back all layer results
    let mut results = Vec::new();
    for i in 1..=num_layers as usize {
        let (num_elems, buffer) = &layer_buffers[i];
        let output_data = unsafe {
            std::slice::from_raw_parts(
                buffer.contents() as *const u32,
                num_elems * 8,
            )
        };

        let mut layer_result = Vec::with_capacity(*num_elems);
        for j in 0..*num_elems {
            let hash_u32s: [u32; 8] = [
                output_data[j * 8],
                output_data[j * 8 + 1],
                output_data[j * 8 + 2],
                output_data[j * 8 + 3],
                output_data[j * 8 + 4],
                output_data[j * 8 + 5],
                output_data[j * 8 + 6],
                output_data[j * 8 + 7],
            ];
            let hash_bytes: [u8; 32] = bytemuck::cast(hash_u32s);
            layer_result.push(Blake2sHash(hash_bytes));
        }
        results.push(layer_result);
    }

    results
}

// Blake2s hash columns are just Vec<Blake2sHash> like in SIMD backend
impl ColumnOps<Blake2sHash> for MetalBackend {
    type Column = Vec<Blake2sHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for MetalBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Blake2sHash> {
        let _timer = crate::metal_profile_fn!("merkle_blake2s", "CPU/GPU", log_size = log_size, num_columns = columns.len());

        // Fall back to SIMD for small sizes or when columns are present
        // (column hashing not yet implemented in Metal)
        if log_size < MIN_MERKLE_LOG_SIZE || !columns.is_empty() || prev_layer.is_none() {
            use crate::prover::backend::Column;
            use crate::prover::backend::simd::column::BaseColumn;

            // Convert Metal columns to SIMD
            let simd_columns_owned: Vec<BaseColumn> = columns
                .iter()
                .map(|col| {
                    let cpu_vals = col.to_cpu();
                    cpu_vals.into_iter().collect()
                })
                .collect();
            let simd_columns_refs: Vec<&BaseColumn> = simd_columns_owned.iter().collect();

            return <SimdBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
                log_size,
                prev_layer,
                &simd_columns_refs,
            );
        }

        // Metal GPU path - hash pairs of child nodes
        let prev_layer = prev_layer.unwrap();
        let num_parents = 1 << log_size;
        assert_eq!(prev_layer.len(), num_parents * 2);

        let ctx = MetalContext::global();
        let device = ctx.device();

        // Flatten children hashes into u32 array
        // Each Blake2sHash is 32 bytes = 8 u32s
        let mut children_data = Vec::with_capacity(prev_layer.len() * 8);
        for hash in prev_layer {
            let hash_u32s: &[u32; 8] = bytemuck::cast_ref(&hash.0);
            children_data.extend_from_slice(hash_u32s);
        }

        // Create buffers
        let children_buffer = device.new_buffer_with_data(
            children_data.as_ptr() as *const _,
            (children_data.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Allocate output buffer using buffer pool (GPU will write all values)
        let parents_size = (num_parents * 8 * std::mem::size_of::<u32>()) as u64;
        let parents_pooled = ctx.checkout_shared_buffer(parents_size);
        let parents_buffer = parents_pooled.buffer();

        let is_m31_output: bool = false;
        let size_param = num_parents as u32;

        // Dispatch kernel
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.merkle_pipeline());
        encoder.set_buffer(0, Some(&children_buffer), 0);
        encoder.set_buffer(1, Some(&parents_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<bool>() as u64,
            &is_m31_output as *const bool as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &size_param as *const u32 as *const _,
        );

        let num_threads = num_parents as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: threadgroups,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back results
        let output_data = unsafe {
            std::slice::from_raw_parts(
                parents_buffer.contents() as *const u32,
                num_parents * 8,
            )
        };

        // Convert u32 array back to Blake2sHash
        let mut result = Vec::with_capacity(num_parents);
        for i in 0..num_parents {
            let hash_u32s: [u32; 8] = [
                output_data[i * 8],
                output_data[i * 8 + 1],
                output_data[i * 8 + 2],
                output_data[i * 8 + 3],
                output_data[i * 8 + 4],
                output_data[i * 8 + 5],
                output_data[i * 8 + 6],
                output_data[i * 8 + 7],
            ];
            let hash_bytes: [u8; 32] = bytemuck::cast(hash_u32s);
            result.push(Blake2sHash(hash_bytes));
        }

        // Keep pooled buffer alive until after GPU completes and data is read
        drop(parents_pooled);

        result
    }
}

impl MerkleOps<Blake2sM31MerkleHasher> for MetalBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Blake2sHash> {
        let _timer = crate::metal_profile_fn!("merkle_m31", "CPU/GPU", log_size = log_size, num_columns = columns.len());

        // Fall back to SIMD for small sizes
        if log_size < MIN_MERKLE_LOG_SIZE {
            use crate::prover::backend::Column;
            use crate::prover::backend::simd::column::BaseColumn;

            // Convert Metal columns to SIMD
            let simd_columns_owned: Vec<BaseColumn> = columns
                .iter()
                .map(|col| {
                    let cpu_vals = col.to_cpu();
                    cpu_vals.into_iter().collect()
                })
                .collect();
            let simd_columns_refs: Vec<&BaseColumn> = simd_columns_owned.iter().collect();

            return <SimdBackend as MerkleOps<Blake2sM31MerkleHasher>>::commit_on_layer(
                log_size,
                prev_layer,
                &simd_columns_refs,
            );
        }

        let ctx = MetalContext::global();
        let device = ctx.device();
        let domain_size = 1 << log_size;

        // GPU leaf hashing path (when columns are present)
        if !columns.is_empty() {
            assert!(prev_layer.is_none(), "Leaf layer should have no prev_layer");

            // Flatten columns using GPU blit
            let num_columns = columns.len();
            let columns_buffer_size = (num_columns * domain_size * std::mem::size_of::<u32>()) as u64;
            let columns_pooled = ctx.checkout_shared_buffer(columns_buffer_size);
            let columns_buffer = columns_pooled.buffer();

            let command_buffer = ctx.command_queue().new_command_buffer();
            let blit_encoder = command_buffer.new_blit_command_encoder();
            let mut offset = 0u64;
            let col_size = (domain_size * std::mem::size_of::<u32>()) as u64;
            for col in columns {
                blit_encoder.copy_from_buffer(col.buffer(), 0, &columns_buffer, offset, col_size);
                offset += col_size;
            }
            blit_encoder.end_encoding();

            // Leaf hash kernel
            let output_size = (domain_size * 8 * std::mem::size_of::<u32>()) as u64;
            let output_pooled = ctx.checkout_shared_buffer(output_size);
            let output_buffer = output_pooled.buffer();

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(ctx.merkle_leaf_pipeline());
            encoder.set_buffer(0, Some(&columns_buffer), 0);
            let num_columns_u32 = num_columns as u32;
            encoder.set_bytes(1, std::mem::size_of::<u32>() as u64, &num_columns_u32 as *const u32 as *const _);
            let domain_size_u32 = domain_size as u32;
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &domain_size_u32 as *const u32 as *const _);
            let is_m31_output = true;
            encoder.set_bytes(3, std::mem::size_of::<bool>() as u64, &is_m31_output as *const bool as *const _);
            encoder.set_buffer(4, Some(&output_buffer), 0);

            let num_threads = domain_size as u64;
            let threadgroup_size = 256.min(num_threads.max(1));
            let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;
            encoder.dispatch_thread_groups(
                metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
                metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let output_data = unsafe { std::slice::from_raw_parts(output_buffer.contents() as *const u32, domain_size * 8) };
            let mut result = Vec::with_capacity(domain_size);
            for i in 0..domain_size {
                let hash_u32s: [u32; 8] = output_data[i * 8..(i + 1) * 8].try_into().unwrap();
                // Convert u32 array to u8 array (little-endian)
                let mut hash_u8s = [0u8; 32];
                for (j, &val) in hash_u32s.iter().enumerate() {
                    let bytes = val.to_le_bytes();
                    hash_u8s[j * 4..(j + 1) * 4].copy_from_slice(&bytes);
                }
                result.push(Blake2sHash(hash_u8s));
            }
            drop(columns_pooled);
            drop(output_pooled);
            return result;
        }

        // Metal GPU path with M31 reduction (node hashing)
        let prev_layer = prev_layer.unwrap();
        let num_parents = domain_size;
        assert_eq!(prev_layer.len(), num_parents * 2);

        // Flatten children hashes into u32 array
        let mut children_data = Vec::with_capacity(prev_layer.len() * 8);
        for hash in prev_layer {
            let hash_u32s: &[u32; 8] = bytemuck::cast_ref(&hash.0);
            children_data.extend_from_slice(hash_u32s);
        }

        // Create buffers
        let children_buffer = device.new_buffer_with_data(
            children_data.as_ptr() as *const _,
            (children_data.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Allocate output buffer using buffer pool (GPU will write all values)
        let parents_size = (num_parents * 8 * std::mem::size_of::<u32>()) as u64;
        let parents_pooled = ctx.checkout_shared_buffer(parents_size);
        let parents_buffer = parents_pooled.buffer();

        let is_m31_output: bool = true; // M31 reduction enabled
        let size_param = num_parents as u32;

        // Dispatch kernel
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.merkle_pipeline());
        encoder.set_buffer(0, Some(&children_buffer), 0);
        encoder.set_buffer(1, Some(&parents_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<bool>() as u64,
            &is_m31_output as *const bool as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &size_param as *const u32 as *const _,
        );

        let num_threads = num_parents as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: threadgroups,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back results
        let output_data = unsafe {
            std::slice::from_raw_parts(
                parents_buffer.contents() as *const u32,
                num_parents * 8,
            )
        };

        // Convert u32 array back to Blake2sHash (already M31-reduced by GPU)
        let mut result = Vec::with_capacity(num_parents);
        for i in 0..num_parents {
            let hash_u32s: [u32; 8] = [
                output_data[i * 8],
                output_data[i * 8 + 1],
                output_data[i * 8 + 2],
                output_data[i * 8 + 3],
                output_data[i * 8 + 4],
                output_data[i * 8 + 5],
                output_data[i * 8 + 6],
                output_data[i * 8 + 7],
            ];
            let hash_bytes: [u8; 32] = bytemuck::cast(hash_u32s);
            result.push(Blake2sHash(hash_bytes));
        }

        // Keep pooled buffer alive until after GPU completes and data is read
        drop(parents_pooled);

        result
    }
}
