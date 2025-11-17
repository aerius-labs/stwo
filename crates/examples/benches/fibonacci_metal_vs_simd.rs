//! End-to-end proof generation benchmark: Metal GPU vs SIMD CPU
//!
//! This benchmark compares the core operations in STARK proof generation
//! between Metal and SIMD backends. Since full proof generation is complex,
//! we focus on the operations that dominate proving time:
//! - FFT (forward transform during commitment)
//! - IFFT (interpolation for quotient computation)
//! - FRI folding
//!
//! Run with: cargo bench --features metal_prover --bench fibonacci_metal_vs_simd

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Col;
use stwo::prover::poly::circle::PolyOps;

#[cfg(target_os = "macos")]
use stwo::prover::backend::metal::MetalBackend;

/// Benchmark the full workflow: coefficient -> evaluation (FFT) -> interpolation (IFFT)
/// This simulates the core operations during trace commitment and quotient computation
fn bench_simd_fft_ifft_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_ifft_roundtrip_simd");
    group.sample_size(20);

    // Test problem sizes from small to large
    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size * 2)); // Count both FFT and IFFT

        group.bench_function(BenchmarkId::from_parameter(log_size), |b| {
            // Setup
            let domain = CanonicCoset::new(log_size).circle_domain();
            let twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
            let coeffs: Col<SimdBackend, BaseField> =
                (0..size).map(|i| BaseField::from(i as u32)).collect();

            b.iter(|| {
                // This is the core workflow during proving:
                // 1. Start with coefficients (trace or quotient polynomial)
                let poly = black_box(stwo::prover::poly::circle::CircleCoefficients::new(
                    coeffs.clone(),
                ));

                // 2. Evaluate for commitment (FFT)
                let eval = SimdBackend::evaluate(&poly, domain, &twiddles);

                // 3. Interpolate for next step (IFFT)
                let _poly_back = SimdBackend::interpolate(eval, &twiddles);
            });
        });
    }

    group.finish();
}

#[cfg(target_os = "macos")]
fn bench_metal_fft_ifft_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_ifft_roundtrip_metal");
    group.sample_size(20);

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size * 2));

        group.bench_function(BenchmarkId::from_parameter(log_size), |b| {
            // Setup
            let domain = CanonicCoset::new(log_size).circle_domain();
            let twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
            let coeffs: Col<MetalBackend, BaseField> =
                (0..size).map(|i| BaseField::from(i as u32)).collect();

            b.iter(|| {
                // Same workflow as SIMD
                let poly = black_box(stwo::prover::poly::circle::CircleCoefficients::new(
                    coeffs.clone(),
                ));

                let eval = MetalBackend::evaluate(&poly, domain, &twiddles);

                let _poly_back = MetalBackend::interpolate(eval, &twiddles);
            });
        });
    }

    group.finish();
}

/// Benchmark just the evaluation (FFT) operation
fn bench_simd_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation_simd");
    group.sample_size(20);

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size));

        group.bench_function(BenchmarkId::from_parameter(log_size), |b| {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
            let coeffs: Col<SimdBackend, BaseField> =
                (0..size).map(|i| BaseField::from(i as u32)).collect();
            let poly = stwo::prover::poly::circle::CircleCoefficients::new(coeffs);

            b.iter(|| {
                let _eval = black_box(SimdBackend::evaluate(&poly, domain, &twiddles));
            });
        });
    }

    group.finish();
}

#[cfg(target_os = "macos")]
fn bench_metal_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation_metal");
    group.sample_size(20);

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size));

        group.bench_function(BenchmarkId::from_parameter(log_size), |b| {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
            let coeffs: Col<MetalBackend, BaseField> =
                (0..size).map(|i| BaseField::from(i as u32)).collect();
            let poly = stwo::prover::poly::circle::CircleCoefficients::new(coeffs);

            b.iter(|| {
                let _eval = black_box(MetalBackend::evaluate(&poly, domain, &twiddles));
            });
        });
    }

    group.finish();
}

#[cfg(target_os = "macos")]
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_simd_fft_ifft_roundtrip, bench_metal_fft_ifft_roundtrip,
              bench_simd_evaluation, bench_metal_evaluation
);

#[cfg(not(target_os = "macos"))]
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_simd_fft_ifft_roundtrip, bench_simd_evaluation
);

criterion_main!(benches);
