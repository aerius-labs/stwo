//! Metal compute kernels for Stwo prover operations.
//!
//! All kernels operate on the Mersenne-31 prime field (p = 2^31 - 1).
//! Field elements are represented as uint32_t in range [0, 2^31 - 1].

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants and Field Arithmetic
// ============================================================================

constant uint32_t M31_PRIME = 0x7FFFFFFF;  // 2^31 - 1

/// Reduce a uint32_t to the range [0, M31_PRIME).
inline uint32_t m31_reduce(uint32_t x) {
    // Since 2^31 ≡ 1 (mod 2^31 - 1), we can reduce by:
    // x = (x & M31_PRIME) + (x >> 31)
    uint32_t reduced = (x & M31_PRIME) + (x >> 31);
    // May need one more reduction if reduced == M31_PRIME
    return reduced == M31_PRIME ? 0 : reduced;
}

/// Add two M31 field elements.
inline uint32_t m31_add(uint32_t a, uint32_t b) {
    return m31_reduce(a + b);
}

/// Subtract two M31 field elements.
inline uint32_t m31_sub(uint32_t a, uint32_t b) {
    // Note: M31_PRIME + a - b is always non-negative
    return m31_reduce(M31_PRIME + a - b);
}

/// Multiply two M31 field elements.
inline uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t product = uint64_t(a) * uint64_t(b);
    // Reduce: product mod (2^31 - 1)
    uint32_t low = uint32_t(product) & M31_PRIME;
    uint32_t high = uint32_t(product >> 31);
    return m31_reduce(low + high);
}

/// Multiply M31 element by a doubled twiddle factor.
/// Twiddles are stored as 2*value for optimization.
inline uint32_t m31_mul_twiddle_dbl(uint32_t a, uint32_t b_dbl) {
    uint64_t product = uint64_t(a) * uint64_t(b_dbl);
    // Since b_dbl = 2*b, product = 2*a*b
    // Shift right by 1 to get a*b, then reduce
    uint64_t shifted = product >> 1;
    uint32_t low = uint32_t(shifted) & M31_PRIME;
    uint32_t high = uint32_t(shifted >> 31);
    return m31_reduce(low + high);
}

// ============================================================================
// FFT Butterfly Operations
// ============================================================================

/// Forward FFT butterfly operation.
/// Computes: (v0 + t*v1, v0 - t*v1)
inline void fft_butterfly(
    thread uint32_t& v0,
    thread uint32_t& v1,
    uint32_t twiddle_dbl
) {
    uint32_t prod = m31_mul_twiddle_dbl(v1, twiddle_dbl);
    uint32_t sum = m31_add(v0, prod);
    uint32_t diff = m31_sub(v0, prod);
    v0 = sum;
    v1 = diff;
}

/// Inverse FFT butterfly operation.
/// Computes: (v0 + v1, (v0 - v1) * t)
inline void ifft_butterfly(
    thread uint32_t& v0,
    thread uint32_t& v1,
    uint32_t twiddle_dbl
) {
    uint32_t sum = m31_add(v0, v1);
    uint32_t diff = m31_sub(v0, v1);
    uint32_t prod = m31_mul_twiddle_dbl(diff, twiddle_dbl);
    v0 = sum;
    v1 = prod;
}

// ============================================================================
// FFT Kernels
// ============================================================================

/// Circle FFT Radix-8 kernel.
///
/// Each thread processes 8 elements through 3 butterfly layers (radix-8).
/// This implements a Cooley-Tukey decimation-in-frequency FFT adapted for
/// the circle group using bit-reversed twiddle factors.
///
/// Parameters:
/// - data: Input/output buffer in natural order (input) / bit-reversed order (output)
/// - twiddles_layer0/1/2: Doubled twiddle factors for each layer (bit-reversed)
/// - log_size: log2(FFT size)
/// - log_step: Current layer offset
/// - layer: Starting layer index (layer, layer+1, layer+2 are processed)
kernel void circle_fft_radix8(
    device uint32_t* data [[buffer(0)]],
    device const uint32_t* twiddles_layer0 [[buffer(1)]],
    device const uint32_t* twiddles_layer1 [[buffer(2)]],
    device const uint32_t* twiddles_layer2 [[buffer(3)]],
    constant uint32_t& log_size [[buffer(4)]],
    constant uint32_t& layer [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread processes 8 elements with stride 2^layer
    // Thread gid processes starting position gid
    uint32_t offset = gid;
    uint32_t stride = 1u << layer;

    // Compute index for twiddle lookup (which block of size 2^(layer+3) we're in)
    // SIMD uses: offset = index << (layer + 3), so index = position >> (layer + 3)
    uint32_t index = gid >> (layer + 3);

    // Load 8 elements with stride (matching SIMD fft3 implementation)
    uint32_t v0 = data[offset + (0 << layer)];
    uint32_t v1 = data[offset + (1 << layer)];
    uint32_t v2 = data[offset + (2 << layer)];
    uint32_t v3 = data[offset + (3 << layer)];
    uint32_t v4 = data[offset + (4 << layer)];
    uint32_t v5 = data[offset + (5 << layer)];
    uint32_t v6 = data[offset + (6 << layer)];
    uint32_t v7 = data[offset + (7 << layer)];

    // Layer 2: 4 butterflies (coarsest, stride = 4)
    // SIMD uses: twiddle_dbl[2][(index + i) & mask]
    // Twiddle array has length 2^(layer+2), mask is 2^(layer+2) - 1
    uint32_t tw2 = twiddles_layer2[index & ((1u << (layer + 2)) - 1)];
    fft_butterfly(v0, v4, tw2);
    fft_butterfly(v1, v5, tw2);
    fft_butterfly(v2, v6, tw2);
    fft_butterfly(v3, v7, tw2);

    // Layer 1: 4 butterflies (middle, stride = 2)
    // SIMD uses: twiddle_dbl[1][(index * 2 + i) & mask]
    // Twiddle array has length 2^(layer+1), mask is 2^(layer+1) - 1
    uint32_t tw1_0 = twiddles_layer1[(index * 2 + 0) & ((1u << (layer + 1)) - 1)];
    uint32_t tw1_1 = twiddles_layer1[(index * 2 + 1) & ((1u << (layer + 1)) - 1)];
    fft_butterfly(v0, v2, tw1_0);
    fft_butterfly(v1, v3, tw1_0);
    fft_butterfly(v4, v6, tw1_1);
    fft_butterfly(v5, v7, tw1_1);

    // Layer 0: 4 butterflies (finest, stride = 1)
    // SIMD uses: twiddle_dbl[0][(index * 4 + i) & mask]
    // Twiddle array has length 2^layer, mask is 2^layer - 1
    uint32_t tw0_0 = twiddles_layer0[(index * 4 + 0) & ((1u << layer) - 1)];
    uint32_t tw0_1 = twiddles_layer0[(index * 4 + 1) & ((1u << layer) - 1)];
    uint32_t tw0_2 = twiddles_layer0[(index * 4 + 2) & ((1u << layer) - 1)];
    uint32_t tw0_3 = twiddles_layer0[(index * 4 + 3) & ((1u << layer) - 1)];
    fft_butterfly(v0, v1, tw0_0);
    fft_butterfly(v2, v3, tw0_1);
    fft_butterfly(v4, v5, tw0_2);
    fft_butterfly(v6, v7, tw0_3);

    // Store results with stride
    data[offset + (0 << layer)] = v0;
    data[offset + (1 << layer)] = v1;
    data[offset + (2 << layer)] = v2;
    data[offset + (3 << layer)] = v3;
    data[offset + (4 << layer)] = v4;
    data[offset + (5 << layer)] = v5;
    data[offset + (6 << layer)] = v6;
    data[offset + (7 << layer)] = v7;
}

/// Circle IFFT Radix-8 kernel.
///
/// Each thread processes 8 elements through 3 inverse butterfly layers.
/// This implements decimation-in-time inverse FFT for the circle group.
kernel void circle_ifft_radix8(
    device uint32_t* data [[buffer(0)]],
    device const uint32_t* itwiddles_layer0 [[buffer(1)]],
    device const uint32_t* itwiddles_layer1 [[buffer(2)]],
    device const uint32_t* itwiddles_layer2 [[buffer(3)]],
    constant uint32_t& log_size [[buffer(4)]],
    constant uint32_t& layer [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread processes 8 elements with stride 2^layer
    // Thread gid processes starting position gid
    uint32_t offset = gid;
    uint32_t stride = 1u << layer;

    // Compute index for twiddle lookup (which block of size 2^(layer+3) we're in)
    uint32_t index = gid >> (layer + 3);

    // Load 8 elements with stride (matching SIMD ifft3 implementation)
    uint32_t v0 = data[offset + (0 << layer)];
    uint32_t v1 = data[offset + (1 << layer)];
    uint32_t v2 = data[offset + (2 << layer)];
    uint32_t v3 = data[offset + (3 << layer)];
    uint32_t v4 = data[offset + (4 << layer)];
    uint32_t v5 = data[offset + (5 << layer)];
    uint32_t v6 = data[offset + (6 << layer)];
    uint32_t v7 = data[offset + (7 << layer)];

    // Layer 0: 4 inverse butterflies (finest, stride = 1)
    // SIMD uses: twiddle_dbl[0][(index * 4 + i) & mask]
    // Twiddle array has length 2^layer, mask is 2^layer - 1
    uint32_t itw0_0 = itwiddles_layer0[(index * 4 + 0) & ((1u << layer) - 1)];
    uint32_t itw0_1 = itwiddles_layer0[(index * 4 + 1) & ((1u << layer) - 1)];
    uint32_t itw0_2 = itwiddles_layer0[(index * 4 + 2) & ((1u << layer) - 1)];
    uint32_t itw0_3 = itwiddles_layer0[(index * 4 + 3) & ((1u << layer) - 1)];
    ifft_butterfly(v0, v1, itw0_0);
    ifft_butterfly(v2, v3, itw0_1);
    ifft_butterfly(v4, v5, itw0_2);
    ifft_butterfly(v6, v7, itw0_3);

    // Layer 1: 4 inverse butterflies (middle, stride = 2)
    // SIMD uses: twiddle_dbl[1][(index * 2 + i) & mask]
    // Twiddle array has length 2^(layer+1), mask is 2^(layer+1) - 1
    uint32_t itw1_0 = itwiddles_layer1[(index * 2 + 0) & ((1u << (layer + 1)) - 1)];
    uint32_t itw1_1 = itwiddles_layer1[(index * 2 + 1) & ((1u << (layer + 1)) - 1)];
    ifft_butterfly(v0, v2, itw1_0);
    ifft_butterfly(v1, v3, itw1_0);
    ifft_butterfly(v4, v6, itw1_1);
    ifft_butterfly(v5, v7, itw1_1);

    // Layer 2: 4 inverse butterflies (coarsest, stride = 4)
    // SIMD uses: twiddle_dbl[2][(index + i) & mask]
    // Twiddle array has length 2^(layer+2), mask is 2^(layer+2) - 1
    uint32_t itw2 = itwiddles_layer2[index & ((1u << (layer + 2)) - 1)];
    ifft_butterfly(v0, v4, itw2);
    ifft_butterfly(v1, v5, itw2);
    ifft_butterfly(v2, v6, itw2);
    ifft_butterfly(v3, v7, itw2);

    // Store results with stride
    data[offset + (0 << layer)] = v0;
    data[offset + (1 << layer)] = v1;
    data[offset + (2 << layer)] = v2;
    data[offset + (3 << layer)] = v3;
    data[offset + (4 << layer)] = v4;
    data[offset + (5 << layer)] = v5;
    data[offset + (6 << layer)] = v6;
    data[offset + (7 << layer)] = v7;
}

// ============================================================================
// FRI Kernels
// ============================================================================

/// FRI fold over circle (placeholder).
kernel void fri_fold_circle(
    device const uint32_t* layer_in [[buffer(0)]],
    device uint32_t* layer_out [[buffer(1)]],
    device const uint32_t* alpha [[buffer(2)]],
    constant uint32_t& log_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Placeholder implementation
}

/// FRI fold over line (placeholder).
kernel void fri_fold_line(
    device const uint32_t* layer_in [[buffer(0)]],
    device uint32_t* layer_out [[buffer(1)]],
    device const uint32_t* alpha [[buffer(2)]],
    constant uint32_t& log_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Placeholder implementation
}

// ============================================================================
// Quotient Accumulation Kernels
// ============================================================================

/// Quotient accumulation kernel (placeholder).
kernel void quotient_accumulate(
    device const uint32_t* numerator [[buffer(0)]],
    device const uint32_t* denominator [[buffer(1)]],
    device uint32_t* accumulator [[buffer(2)]],
    device const uint32_t* random_coeff [[buffer(3)]],
    constant uint32_t& size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Placeholder implementation
}

// ============================================================================
// Merkle Kernels
// ============================================================================

/// Merkle tree BLAKE2s hashing kernel (placeholder).
kernel void merkle_blake2s(
    device const uint32_t* leaves [[buffer(0)]],
    device uint32_t* nodes [[buffer(1)]],
    constant uint32_t& layer [[buffer(2)]],
    constant uint32_t& log_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Placeholder implementation
}

// ============================================================================
// Lookup (MLE) Kernels
// ============================================================================

/// MLE fold kernel for GKR lookups (placeholder).
kernel void mle_fold(
    device const uint32_t* mle_in [[buffer(0)]],
    device uint32_t* mle_out [[buffer(1)]],
    device const uint32_t* challenge [[buffer(2)]],
    constant uint32_t& log_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Placeholder implementation
}
