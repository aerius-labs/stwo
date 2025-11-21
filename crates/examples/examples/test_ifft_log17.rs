use stwo::core::poly::circle::CanonicCoset;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::metal::MetalBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::metal::MetalBaseColumn;

fn main() {
    let log_size = 17u32;
    println!("\n=== Testing IFFT at log_size {} ===", log_size);
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Simple pattern
    let values: Vec<BaseField> = (0..domain.size())
        .map(|i| BaseField::from_u32_unchecked((i % 256) as u32))
        .collect();

    println!("Input values[0..16]: {:?}", &values[0..16].iter().map(|v| v.0).collect::<Vec<_>>());

    let eval_simd = CircleEvaluation::new(domain, BaseColumn::from_iter(values.clone()));
    let eval_metal = CircleEvaluation::new(domain, MetalBaseColumn::from_iter(values));

    let twiddles_simd = SimdBackend::precompute_twiddles(domain.half_coset);
    let twiddles_metal = MetalBackend::precompute_twiddles(domain.half_coset);

    println!("\nRunning SIMD interpolate...");
    let poly_simd = SimdBackend::interpolate(eval_simd, &twiddles_simd);

    println!("\nRunning Metal interpolate...");
    let poly_metal = MetalBackend::interpolate(eval_metal, &twiddles_metal);

    let simd_coeffs = poly_simd.coeffs.to_cpu();
    let metal_coeffs = poly_metal.coeffs.to_cpu();

    println!("\nComparing results:");
    let mut mismatch_count = 0;
    for i in 0..64 {
        if simd_coeffs[i] != metal_coeffs[i] {
            if mismatch_count < 10 {
                println!("MISMATCH at index {}: SIMD={}, Metal={}",
                         i, simd_coeffs[i].0, metal_coeffs[i].0);
            }
            mismatch_count += 1;
        }
    }
    println!("Total mismatches in first 64: {}", mismatch_count);

    println!("\nSIMD coeffs[0..32]: {:?}", &simd_coeffs[0..32].iter().map(|v| v.0).collect::<Vec<_>>());
    println!("Metal coeffs[0..32]: {:?}", &metal_coeffs[0..32].iter().map(|v| v.0).collect::<Vec<_>>());
}
