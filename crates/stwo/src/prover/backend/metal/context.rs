//! Metal context and device management.
//!
//! This module provides the central `MetalContext` singleton that manages:
//! - Metal device and command queue
//! - Pre-compiled compute pipeline states
//! - Shared memory buffer allocations
//!
//! The context is lazily initialized on first use and can be safely shared
//! across threads using `MetalContextHandle`.

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::core::poly::circle::CircleDomain;
use crate::core::utils::bit_reverse_index;

use super::shaders;
use super::twiddle_manager::FlatTwiddleManager;
use super::buffer_pool::GlobalPools;

/// Global Metal context singleton.
static METAL_CONTEXT: OnceLock<Arc<MetalContext>> = OnceLock::new();

/// Initialize profiling on first context access.
fn init_profiling_once() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        super::profiling::init_profiling();
    });
}

/// Metal context managing device, queue, and pipeline states.
///
/// This is the central coordination point for all Metal GPU operations.
/// It holds pre-compiled compute pipelines and provides command buffer creation.
pub struct MetalContext {
    /// Metal device (GPU).
    device: Device,

    /// Command queue for submitting work.
    command_queue: CommandQueue,

    /// Metal shader library.
    #[allow(dead_code)]
    library: Library,

    /// FFT radix-8 kernel pipeline.
    fft_radix8_pipeline: ComputePipelineState,

    /// IFFT radix-8 kernel pipeline.
    ifft_radix8_pipeline: ComputePipelineState,

    /// FFT radix-2 kernel pipeline (for vecwise layers).
    fft_radix2_pipeline: ComputePipelineState,

    /// IFFT radix-2 kernel pipeline (for vecwise layers).
    ifft_radix2_pipeline: ComputePipelineState,

    /// Fused vecwise FFT kernel pipeline (layers 1-4).
    fft_vecwise_fused_pipeline: ComputePipelineState,

    /// IFFT normalization kernel pipeline.
    ifft_normalize_pipeline: ComputePipelineState,

    /// FRI fold (circle) kernel pipeline.
    fri_fold_circle_pipeline: ComputePipelineState,

    /// FRI fold (line) kernel pipeline.
    fri_fold_line_pipeline: ComputePipelineState,

    /// Quotient accumulation kernel pipeline.
    quotient_pipeline: ComputePipelineState,

    /// Merkle BLAKE2s kernel pipeline.
    merkle_pipeline: ComputePipelineState,

    /// MLE fold M31→QM31 kernel pipeline (for lookups).
    mle_fold_m31_pipeline: ComputePipelineState,

    /// MLE fold QM31→QM31 kernel pipeline (for lookups).
    mle_fold_qm31_pipeline: ComputePipelineState,

    /// Proof-of-work grinding kernel pipeline.
    grind_pipeline: ComputePipelineState,

    /// Pack coordinates to QM31 interleaved format kernel pipeline.
    pack_coords_to_qm31_pipeline: ComputePipelineState,

    /// Unpack QM31 interleaved format to coordinates kernel pipeline.
    unpack_qm31_to_coords_pipeline: ComputePipelineState,

    /// Cache for twiddle factor buffers.
    /// Key is a hash of the twiddle data, value is the Metal buffer.
    twiddle_cache: Mutex<HashMap<u64, Buffer>>,

    /// Manager for flattened twiddle buffers.
    flat_twiddle_manager: FlatTwiddleManager,

    /// Global buffer pools for reusable buffers.
    buffer_pools: GlobalPools,

    /// Cache for domain evaluation points (x, y) in bit-reversed order.
    /// Key is log_size, value is (x_buffer, y_buffer) with M31 values.
    domain_xy_cache: Mutex<HashMap<u32, (Buffer, Buffer)>>,
}

impl MetalContext {
    /// Initialize Metal context with device and compile all pipelines.
    ///
    /// # Panics
    /// Panics if Metal device is not available or shader compilation fails.
    fn new() -> Result<Self, String> {
        // Get default Metal device
        let device = Device::system_default()
            .ok_or_else(|| "No Metal device found".to_string())?;

        // Create command queue
        let command_queue = device.new_command_queue();

        // Load shader library
        let shader_source = shaders::KERNEL_SOURCE;
        let compile_options = CompileOptions::new();

        #[cfg(feature = "metal_debug")]
        {
            compile_options.set_fast_math_enabled(false);
        }
        #[cfg(not(feature = "metal_debug"))]
        {
            compile_options.set_fast_math_enabled(true);
        }

        let library = device
            .new_library_with_source(shader_source, &compile_options)
            .map_err(|e| format!("Failed to compile Metal shaders: {}", e))?;

        // Compile all pipeline states
        let fft_radix8_pipeline = Self::create_pipeline(&device, &library, "circle_fft_radix8")?;
        let ifft_radix8_pipeline = Self::create_pipeline(&device, &library, "circle_ifft_radix8")?;
        let fft_radix2_pipeline = Self::create_pipeline(&device, &library, "circle_fft_radix2")?;
        let ifft_radix2_pipeline = Self::create_pipeline(&device, &library, "circle_ifft_radix2")?;
        let fft_vecwise_fused_pipeline = Self::create_pipeline(&device, &library, "circle_fft_vecwise_fused")?;
        let ifft_normalize_pipeline = Self::create_pipeline(&device, &library, "ifft_normalize_m31")?;
        let fri_fold_circle_pipeline =
            Self::create_pipeline(&device, &library, "fri_fold_circle_into_line")?;
        let fri_fold_line_pipeline = Self::create_pipeline(&device, &library, "fri_fold_line")?;
        let quotient_pipeline = Self::create_pipeline(&device, &library, "quotient_accumulate")?;
        let merkle_pipeline = Self::create_pipeline(&device, &library, "merkle_blake2s")?;
        let mle_fold_m31_pipeline = Self::create_pipeline(&device, &library, "mle_fold_m31_to_qm31")?;
        let mle_fold_qm31_pipeline = Self::create_pipeline(&device, &library, "mle_fold_qm31_to_qm31")?;
        let grind_pipeline = Self::create_pipeline(&device, &library, "grind_pow")?;
        let pack_coords_to_qm31_pipeline = Self::create_pipeline(&device, &library, "pack_coords_to_qm31")?;
        let unpack_qm31_to_coords_pipeline = Self::create_pipeline(&device, &library, "unpack_qm31_to_coords")?;

        let buffer_pools = GlobalPools::new(device.clone());

        Ok(Self {
            device,
            command_queue,
            library,
            fft_radix8_pipeline,
            ifft_radix8_pipeline,
            fft_radix2_pipeline,
            ifft_radix2_pipeline,
            fft_vecwise_fused_pipeline,
            ifft_normalize_pipeline,
            fri_fold_circle_pipeline,
            fri_fold_line_pipeline,
            quotient_pipeline,
            merkle_pipeline,
            mle_fold_m31_pipeline,
            mle_fold_qm31_pipeline,
            grind_pipeline,
            pack_coords_to_qm31_pipeline,
            unpack_qm31_to_coords_pipeline,
            twiddle_cache: Mutex::new(HashMap::new()),
            flat_twiddle_manager: FlatTwiddleManager::new(),
            buffer_pools,
            domain_xy_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Create a compute pipeline state for a kernel function.
    fn create_pipeline(
        device: &Device,
        library: &Library,
        function_name: &str,
    ) -> Result<ComputePipelineState, String> {
        let function = library
            .get_function(function_name, None)
            .map_err(|e| format!("Failed to get function '{}': {}", function_name, e))?;

        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create pipeline for '{}': {}", function_name, e))
    }

    /// Get the global Metal context, initializing it if needed.
    pub fn global() -> Arc<Self> {
        init_profiling_once();
        METAL_CONTEXT
            .get_or_init(|| {
                Arc::new(Self::new().expect("Failed to initialize Metal context"))
            })
            .clone()
    }

    /// Check if Metal is available on this system.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Get the Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the command queue.
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Get FFT radix-8 pipeline.
    pub fn fft_radix8_pipeline(&self) -> &ComputePipelineState {
        &self.fft_radix8_pipeline
    }

    /// Get IFFT radix-8 pipeline.
    pub fn ifft_radix8_pipeline(&self) -> &ComputePipelineState {
        &self.ifft_radix8_pipeline
    }

    /// Get FFT radix-2 pipeline.
    pub fn fft_radix2_pipeline(&self) -> &ComputePipelineState {
        &self.fft_radix2_pipeline
    }

    /// Get IFFT radix-2 pipeline.
    pub fn ifft_radix2_pipeline(&self) -> &ComputePipelineState {
        &self.ifft_radix2_pipeline
    }

    /// Get fused vecwise FFT pipeline (layers 1-4).
    pub fn fft_vecwise_fused_pipeline(&self) -> &ComputePipelineState {
        &self.fft_vecwise_fused_pipeline
    }

    /// Get IFFT normalization pipeline.
    pub fn ifft_normalize_pipeline(&self) -> &ComputePipelineState {
        &self.ifft_normalize_pipeline
    }

    /// Get FRI fold (circle) pipeline.
    pub fn fri_fold_circle_pipeline(&self) -> &ComputePipelineState {
        &self.fri_fold_circle_pipeline
    }

    /// Get FRI fold (line) pipeline.
    pub fn fri_fold_line_pipeline(&self) -> &ComputePipelineState {
        &self.fri_fold_line_pipeline
    }

    /// Get quotient accumulation pipeline.
    pub fn quotient_pipeline(&self) -> &ComputePipelineState {
        &self.quotient_pipeline
    }

    /// Get Merkle hashing pipeline.
    pub fn merkle_pipeline(&self) -> &ComputePipelineState {
        &self.merkle_pipeline
    }

    /// Get MLE fold M31→QM31 pipeline.
    pub fn mle_fold_m31_pipeline(&self) -> &ComputePipelineState {
        &self.mle_fold_m31_pipeline
    }

    /// Get MLE fold QM31→QM31 pipeline.
    pub fn mle_fold_qm31_pipeline(&self) -> &ComputePipelineState {
        &self.mle_fold_qm31_pipeline
    }

    /// Get proof-of-work grinding pipeline.
    pub fn grind_pipeline(&self) -> &ComputePipelineState {
        &self.grind_pipeline
    }

    /// Get pack coordinates to QM31 pipeline.
    pub fn pack_coords_to_qm31_pipeline(&self) -> &ComputePipelineState {
        &self.pack_coords_to_qm31_pipeline
    }

    /// Get unpack QM31 to coordinates pipeline.
    pub fn unpack_qm31_to_coords_pipeline(&self) -> &ComputePipelineState {
        &self.unpack_qm31_to_coords_pipeline
    }

    /// Get or create a flattened twiddle buffer from multiple layers.
    /// Returns the flat buffer and section information for each layer.
    pub fn get_or_create_flat_twiddle_buffer(
        &self,
        twiddle_layers: &[&[u32]],
    ) -> super::twiddle_manager::FlatTwiddleBuffer {
        self.flat_twiddle_manager.get_or_create_flat_buffer(&self.device, twiddle_layers)
    }

    /// Check out a shared memory buffer from the pool.
    pub fn checkout_shared_buffer(&self, size: u64) -> super::buffer_pool::PooledBuffer {
        self.buffer_pools.shared.checkout(size)
    }

    /// Check out a private (GPU-only) memory buffer from the pool.
    #[allow(dead_code)]
    pub fn checkout_private_buffer(&self, size: u64) -> super::buffer_pool::PooledBuffer {
        self.buffer_pools.private.checkout(size)
    }

    /// Get or create a cached twiddle buffer (legacy, single layer).
    /// Uses a simple hash of the twiddle data as the cache key.
    pub fn get_or_create_twiddle_buffer(&self, twiddles: &[u32]) -> Buffer {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Compute a hash of the twiddle data for the cache key
        let mut hasher = DefaultHasher::new();
        twiddles.len().hash(&mut hasher);
        // Sample a few twiddles for the hash (avoid hashing all for large arrays)
        if twiddles.len() > 0 {
            twiddles[0].hash(&mut hasher);
            if twiddles.len() > 1 {
                twiddles[twiddles.len() / 2].hash(&mut hasher);
                twiddles[twiddles.len() - 1].hash(&mut hasher);
            }
        }
        let cache_key = hasher.finish();

        // Check cache first
        {
            let cache = self.twiddle_cache.lock().unwrap();
            if let Some(buffer) = cache.get(&cache_key) {
                return buffer.clone();
            }
        }

        // Create new buffer if not cached
        let buffer = self.device.new_buffer_with_data(
            twiddles.as_ptr() as *const _,
            (twiddles.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Store in cache
        {
            let mut cache = self.twiddle_cache.lock().unwrap();
            cache.insert(cache_key, buffer.clone());
        }

        buffer
    }

    /// Get or create cached domain evaluation point buffers (x, y) in bit-reversed order.
    ///
    /// This caches the expensive domain point computation for reuse across multiple
    /// quotient accumulation calls with the same domain size.
    ///
    /// Returns (x_buffer, y_buffer) where each contains M31 values (u32).
    pub fn get_or_create_domain_xy_buffers(&self, domain: CircleDomain) -> (Buffer, Buffer) {
        let log_size = domain.log_size();
        let cache_key = log_size;

        // Check cache first
        {
            let cache = self.domain_xy_cache.lock().unwrap();
            if let Some((x_buf, y_buf)) = cache.get(&cache_key) {
                return (x_buf.clone(), y_buf.clone());
            }
        }

        // Compute domain points in bit-reversed order
        let domain_size = domain.size();
        let mut domain_points_x = Vec::with_capacity(domain_size);
        let mut domain_points_y = Vec::with_capacity(domain_size);

        for i in 0..domain_size {
            let point = domain.at(bit_reverse_index(i, log_size));
            domain_points_x.push(point.x.0);
            domain_points_y.push(point.y.0);
        }

        // Create Metal buffers
        let x_buffer = self.device.new_buffer_with_data(
            domain_points_x.as_ptr() as *const _,
            (domain_size * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let y_buffer = self.device.new_buffer_with_data(
            domain_points_y.as_ptr() as *const _,
            (domain_size * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Store in cache
        {
            let mut cache = self.domain_xy_cache.lock().unwrap();
            cache.insert(cache_key, (x_buffer.clone(), y_buffer.clone()));
        }

        (x_buffer, y_buffer)
    }

    /// Get MTLResourceOptions for shared memory (unified memory on Apple Silicon).
    ///
    /// This uses MTLStorageModeShared which allows both CPU and GPU to access
    /// the same memory without explicit copies.
    pub fn shared_resource_options() -> MTLResourceOptions {
        MTLResourceOptions::StorageModeShared
    }
}

/// Thread-safe handle to the Metal context.
///
/// This is a lightweight wrapper around Arc<MetalContext> that can be
/// cloned and passed around safely.
#[derive(Clone)]
pub struct MetalContextHandle {
    inner: Arc<MetalContext>,
}

impl MetalContextHandle {
    /// Create a new handle to the global Metal context.
    pub fn new() -> Self {
        Self {
            inner: MetalContext::global(),
        }
    }

    /// Get the underlying Metal context.
    pub fn context(&self) -> &MetalContext {
        &self.inner
    }
}

impl Default for MetalContextHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_context_initialization() {
        // Just check we can create the context without panicking
        if MetalContext::is_available() {
            let handle = MetalContextHandle::new();
            assert!(handle.context().device().name().len() > 0);
        }
    }

    #[test]
    fn test_context_is_singleton() {
        if MetalContext::is_available() {
            let handle1 = MetalContextHandle::new();
            let handle2 = MetalContextHandle::new();

            // Both handles should point to the same context
            assert!(Arc::ptr_eq(&handle1.inner, &handle2.inner));
        }
    }
}
