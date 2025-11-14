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
    CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions,
};
use std::sync::{Arc, OnceLock};

use super::shaders;

/// Global Metal context singleton.
static METAL_CONTEXT: OnceLock<Arc<MetalContext>> = OnceLock::new();

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

    /// FRI fold (circle) kernel pipeline.
    fri_fold_circle_pipeline: ComputePipelineState,

    /// FRI fold (line) kernel pipeline.
    fri_fold_line_pipeline: ComputePipelineState,

    /// Quotient accumulation kernel pipeline.
    quotient_pipeline: ComputePipelineState,

    /// Merkle BLAKE2s kernel pipeline.
    merkle_pipeline: ComputePipelineState,

    /// MLE fold kernel pipeline (for lookups).
    mle_fold_pipeline: ComputePipelineState,
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
        let fri_fold_circle_pipeline =
            Self::create_pipeline(&device, &library, "fri_fold_circle")?;
        let fri_fold_line_pipeline = Self::create_pipeline(&device, &library, "fri_fold_line")?;
        let quotient_pipeline = Self::create_pipeline(&device, &library, "quotient_accumulate")?;
        let merkle_pipeline = Self::create_pipeline(&device, &library, "merkle_blake2s")?;
        let mle_fold_pipeline = Self::create_pipeline(&device, &library, "mle_fold")?;

        Ok(Self {
            device,
            command_queue,
            library,
            fft_radix8_pipeline,
            ifft_radix8_pipeline,
            fft_radix2_pipeline,
            ifft_radix2_pipeline,
            fri_fold_circle_pipeline,
            fri_fold_line_pipeline,
            quotient_pipeline,
            merkle_pipeline,
            mle_fold_pipeline,
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

    /// Get MLE fold pipeline.
    pub fn mle_fold_pipeline(&self) -> &ComputePipelineState {
        &self.mle_fold_pipeline
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
