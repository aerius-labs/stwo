//! Metal-backed column types using shared GPU memory.
//!
//! These column types use `MTLStorageModeShared` which provides unified memory
//! access on Apple Silicon - the same memory is accessible by both CPU and GPU
//! without explicit copies.
//!
//! NOTE: These types are NOT used in Phase 1 (MetalBackend uses SIMD columns).
//! They are experimental types for future Phase 2 optimization.

// Allow dead code since these types are not used in Phase 1
#[allow(dead_code)]

use metal::{Buffer, MTLResourceOptions};
use std::fmt::Debug;

use super::context::MetalContext;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::{Column, CpuBackend};
use crate::prover::backend::simd::SimdBackend;
use crate::prover::secure_column::SecureColumnByCoords;

/// A view into a GPU buffer representing a subrange without copying.
///
/// This allows referencing parts of Metal buffers efficiently for pipeline operations.
#[derive(Clone, Debug)]
pub struct GpuSlice {
    /// The underlying Metal buffer
    pub buffer: Buffer,
    /// Offset in bytes from the start of the buffer
    pub offset_bytes: u64,
    /// Number of elements (not bytes)
    pub len_elems: usize,
}

impl GpuSlice {
    /// Create a new GPU slice
    pub fn new(buffer: Buffer, offset_bytes: u64, len_elems: usize) -> Self {
        Self {
            buffer,
            offset_bytes,
            len_elems,
        }
    }

    /// Get a raw pointer to the data as u32 elements
    pub fn as_u32_ptr(&self) -> (*mut u32, usize) {
        let ptr = unsafe {
            (self.buffer.contents() as *mut u8).add(self.offset_bytes as usize)
        };
        (ptr as *mut u32, self.len_elems)
    }

    /// Get a raw pointer to the data as BaseField elements
    pub fn as_basefield_ptr(&self) -> (*mut BaseField, usize) {
        let ptr = unsafe {
            (self.buffer.contents() as *mut u8).add(self.offset_bytes as usize)
        };
        (ptr as *mut BaseField, self.len_elems)
    }

    /// Get a raw pointer to the data as SecureField elements
    pub fn as_securefield_ptr(&self) -> (*mut SecureField, usize) {
        let ptr = unsafe {
            (self.buffer.contents() as *mut u8).add(self.offset_bytes as usize)
        };
        (ptr as *mut SecureField, self.len_elems)
    }
}

/// Metal-backed column for base field elements (M31).
///
/// Uses shared memory buffer accessible by both CPU and GPU.
#[derive(Clone)]
#[allow(dead_code)]
pub struct MetalBaseColumn {
    /// Metal buffer in shared memory.
    buffer: Buffer,
    /// Number of elements.
    len: usize,
}

impl MetalBaseColumn {
    /// Create a new column from a Metal buffer.
    pub fn from_buffer(buffer: Buffer, len: usize) -> Self {
        Self { buffer, len }
    }

    /// Get the underlying Metal buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get a slice view of the data (CPU-accessible).
    pub fn as_slice(&self) -> &[BaseField] {
        let ptr = self.buffer.contents() as *const BaseField;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Get a mutable slice view of the data (CPU-accessible).
    pub fn as_mut_slice(&mut self) -> &mut [BaseField] {
        let ptr = self.buffer.contents() as *mut BaseField;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }

    /// Get a GPU slice view of this column.
    pub fn as_gpu_slice(&self) -> GpuSlice {
        GpuSlice::new(self.buffer.clone(), 0, self.len)
    }
}

impl Debug for MetalBaseColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBaseColumn")
            .field("len", &self.len)
            .field("data", &self.as_slice())
            .finish()
    }
}

impl Column<BaseField> for MetalBaseColumn {
    fn zeros(len: usize) -> Self {
        let ctx = MetalContext::global();
        let size = len * std::mem::size_of::<BaseField>();
        let buffer = ctx.device().new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Zero the buffer
        let ptr = buffer.contents() as *mut BaseField;
        unsafe {
            std::ptr::write_bytes(ptr, 0, len);
        }

        Self::from_buffer(buffer, len)
    }

    unsafe fn uninitialized(len: usize) -> Self {
        let ctx = MetalContext::global();
        let size = len * std::mem::size_of::<BaseField>();
        let buffer = ctx.device().new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self::from_buffer(buffer, len)
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        self.as_slice().to_vec()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn at(&self, index: usize) -> BaseField {
        self.as_slice()[index]
    }

    fn set(&mut self, index: usize, value: BaseField) {
        self.as_mut_slice()[index] = value;
    }

    fn split_at_mid(self) -> (Self, Self) {
        let mid = self.len / 2;
        let ctx = MetalContext::global();
        let elem_size = std::mem::size_of::<BaseField>();

        // Create two new buffers
        let buffer1 = ctx.device().new_buffer(
            (mid * elem_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer2 = ctx.device().new_buffer(
            (mid * elem_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Copy data
        unsafe {
            let src = self.buffer.contents() as *const BaseField;
            let dst1 = buffer1.contents() as *mut BaseField;
            let dst2 = buffer2.contents() as *mut BaseField;

            std::ptr::copy_nonoverlapping(src, dst1, mid);
            std::ptr::copy_nonoverlapping(src.add(mid), dst2, mid);
        }

        (
            Self::from_buffer(buffer1, mid),
            Self::from_buffer(buffer2, mid),
        )
    }
}

impl FromIterator<BaseField> for MetalBaseColumn {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        let vec: Vec<BaseField> = iter.into_iter().collect();
        let len = vec.len();

        let ctx = MetalContext::global();
        let size = len * std::mem::size_of::<BaseField>();
        let buffer = ctx.device().new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Copy data to buffer
        unsafe {
            let dst = buffer.contents() as *mut BaseField;
            std::ptr::copy_nonoverlapping(vec.as_ptr(), dst, len);
        }

        Self::from_buffer(buffer, len)
    }
}

/// Metal-backed column for secure field elements (QM31).
///
/// SecureField is represented as 4 BaseField elements, so we store
/// it as a flat buffer of BaseField elements with length 4x.
#[derive(Clone)]
#[allow(dead_code)]
pub struct MetalSecureColumn {
    /// Metal buffer in shared memory (stores 4 * len BaseField elements).
    buffer: Buffer,
    /// Number of SecureField elements.
    len: usize,
}

impl MetalSecureColumn {
    /// Create a new column from a Metal buffer.
    pub fn from_buffer(buffer: Buffer, len: usize) -> Self {
        Self { buffer, len }
    }

    /// Get the underlying Metal buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get a slice view of the data (CPU-accessible).
    pub fn as_slice(&self) -> &[SecureField] {
        let ptr = self.buffer.contents() as *const SecureField;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Get a mutable slice view of the data (CPU-accessible).
    pub fn as_mut_slice(&mut self) -> &mut [SecureField] {
        let ptr = self.buffer.contents() as *mut SecureField;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }

    /// Get a GPU slice view of this column.
    pub fn as_gpu_slice(&self) -> GpuSlice {
        GpuSlice::new(self.buffer.clone(), 0, self.len)
    }
}

impl Debug for MetalSecureColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalSecureColumn")
            .field("len", &self.len)
            .field("data", &self.as_slice())
            .finish()
    }
}

impl Column<SecureField> for MetalSecureColumn {
    fn zeros(len: usize) -> Self {
        let ctx = MetalContext::global();
        let size = len * std::mem::size_of::<SecureField>();
        let buffer = ctx.device().new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Zero the buffer
        let ptr = buffer.contents() as *mut SecureField;
        unsafe {
            std::ptr::write_bytes(ptr, 0, len);
        }

        Self::from_buffer(buffer, len)
    }

    unsafe fn uninitialized(len: usize) -> Self {
        let ctx = MetalContext::global();
        let size = len * std::mem::size_of::<SecureField>();
        let buffer = ctx.device().new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self::from_buffer(buffer, len)
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        self.as_slice().to_vec()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn at(&self, index: usize) -> SecureField {
        self.as_slice()[index]
    }

    fn set(&mut self, index: usize, value: SecureField) {
        self.as_mut_slice()[index] = value;
    }

    fn split_at_mid(self) -> (Self, Self) {
        let mid = self.len / 2;
        let ctx = MetalContext::global();
        let elem_size = std::mem::size_of::<SecureField>();

        // Create two new buffers
        let buffer1 = ctx.device().new_buffer(
            (mid * elem_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer2 = ctx.device().new_buffer(
            (mid * elem_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Copy data
        unsafe {
            let src = self.buffer.contents() as *const SecureField;
            let dst1 = buffer1.contents() as *mut SecureField;
            let dst2 = buffer2.contents() as *mut SecureField;

            std::ptr::copy_nonoverlapping(src, dst1, mid);
            std::ptr::copy_nonoverlapping(src.add(mid), dst2, mid);
        }

        (
            Self::from_buffer(buffer1, mid),
            Self::from_buffer(buffer2, mid),
        )
    }
}

impl FromIterator<SecureField> for MetalSecureColumn {
    fn from_iter<T: IntoIterator<Item = SecureField>>(iter: T) -> Self {
        let vec: Vec<SecureField> = iter.into_iter().collect();
        let len = vec.len();

        let ctx = MetalContext::global();
        let size = len * std::mem::size_of::<SecureField>();
        let buffer = ctx.device().new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Copy data to buffer
        unsafe {
            let dst = buffer.contents() as *mut SecureField;
            std::ptr::copy_nonoverlapping(vec.as_ptr(), dst, len);
        }

        Self::from_buffer(buffer, len)
    }
}

// Implementation for SecureColumnByCoords
use super::super::MetalBackend;

impl SecureColumnByCoords<MetalBackend> {
    /// Convert from CPU SecureColumnByCoords to Metal SecureColumnByCoords
    pub fn from_cpu(cpu: SecureColumnByCoords<CpuBackend>) -> Self {
        Self {
            columns: cpu.columns.map(|col| col.into_iter().collect()),
        }
    }

    /// Convert from SIMD SecureColumnByCoords to Metal SecureColumnByCoords
    pub fn from_simd(simd: SecureColumnByCoords<SimdBackend>) -> Self {
        use crate::prover::backend::Column;
        Self {
            columns: simd.columns.map(|col| {
                let cpu_vals = col.to_cpu();
                cpu_vals.into_iter().collect()
            }),
        }
    }

    /// Create an interleaved QM31 buffer (4 u32s per element) from coordinate columns.
    ///
    /// Layout: [a0, b0, c0, d0, a1, b1, c1, d1, ...]
    /// where each QM31 = {CM31(a,b), CM31(c,d)} and each M31 is stored as u32.
    ///
    /// This writes directly into a Metal shared buffer, avoiding intermediate Vec allocations.
    pub fn as_qm31_interleaved_u32_buffer(&self) -> Buffer {
        let len = self.len();
        let ctx = MetalContext::global();
        let device = ctx.device();

        // Allocate output buffer (4 u32s per SecureField element)
        let buffer_size = (len * 4 * std::mem::size_of::<u32>()) as u64;
        let buffer = device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

        // Get pointers to each coordinate column via zero-copy
        let col0_slice = self.columns[0].as_slice();
        let col1_slice = self.columns[1].as_slice();
        let col2_slice = self.columns[2].as_slice();
        let col3_slice = self.columns[3].as_slice();

        // Write interleaved QM31 data directly to buffer
        unsafe {
            let dst = buffer.contents() as *mut u32;
            for i in 0..len {
                *dst.add(i * 4 + 0) = col0_slice[i].0;
                *dst.add(i * 4 + 1) = col1_slice[i].0;
                *dst.add(i * 4 + 2) = col2_slice[i].0;
                *dst.add(i * 4 + 3) = col3_slice[i].0;
            }
        }

        buffer
    }

    /// Create a SecureColumnByCoords from an interleaved QM31 buffer.
    ///
    /// This is the inverse of `as_qm31_interleaved_u32_buffer`.
    pub unsafe fn from_qm31_interleaved_buffer(buffer: &Buffer, len: usize) -> Self {
        let ctx = MetalContext::global();
        let device = ctx.device();

        // Create 4 separate column buffers
        let columns = std::array::from_fn(|_| {
            device.new_buffer(
                (len * std::mem::size_of::<BaseField>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        });

        // De-interleave QM31 buffer into 4 coordinate columns
        let src = buffer.contents() as *const u32;
        for i in 0..len {
            let a = *src.add(i * 4 + 0);
            let b = *src.add(i * 4 + 1);
            let c = *src.add(i * 4 + 2);
            let d = *src.add(i * 4 + 3);

            *(columns[0].contents() as *mut u32).add(i) = a;
            *(columns[1].contents() as *mut u32).add(i) = b;
            *(columns[2].contents() as *mut u32).add(i) = c;
            *(columns[3].contents() as *mut u32).add(i) = d;
        }

        Self {
            columns: columns.map(|buf| MetalBaseColumn::from_buffer(buf, len)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::m31::M31;

    #[test]
    fn test_metal_base_column_zeros() {
        if !MetalContext::is_available() {
            return;
        }

        let col = MetalBaseColumn::zeros(100);
        assert_eq!(col.len(), 100);
        assert_eq!(col.at(0), M31::from(0));
        assert_eq!(col.at(99), M31::from(0));
    }

    #[test]
    fn test_metal_base_column_from_iter() {
        if !MetalContext::is_available() {
            return;
        }

        let data: Vec<BaseField> = (0..10).map(M31::from).collect();
        let col: MetalBaseColumn = data.iter().cloned().collect();

        assert_eq!(col.len(), 10);
        assert_eq!(col.at(0), M31::from(0));
        assert_eq!(col.at(9), M31::from(9));
    }

    #[test]
    fn test_metal_base_column_split() {
        if !MetalContext::is_available() {
            return;
        }

        let data: Vec<BaseField> = (0..8).map(M31::from).collect();
        let col: MetalBaseColumn = data.iter().cloned().collect();

        let (left, right) = col.split_at_mid();

        assert_eq!(left.len(), 4);
        assert_eq!(right.len(), 4);
        assert_eq!(left.at(0), M31::from(0));
        assert_eq!(left.at(3), M31::from(3));
        assert_eq!(right.at(0), M31::from(4));
        assert_eq!(right.at(3), M31::from(7));
    }

    #[test]
    fn test_metal_secure_column_zeros() {
        if !MetalContext::is_available() {
            return;
        }

        let col = MetalSecureColumn::zeros(50);
        assert_eq!(col.len(), 50);
        // Check that zeros are actually zero by converting to u32
        let val0 = col.at(0);
        let val49 = col.at(49);
        assert_eq!(val0, SecureField::from_m31(M31::from(0), M31::from(0), M31::from(0), M31::from(0)));
        assert_eq!(val49, SecureField::from_m31(M31::from(0), M31::from(0), M31::from(0), M31::from(0)));
    }
}
