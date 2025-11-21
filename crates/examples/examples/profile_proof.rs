//! Profile a single Metal proof generation with detailed timing breakdown
//!
//! Run with: METAL_PROFILE=1 cargo run --release --features metal_prover --example profile_proof

use itertools::Itertools;
use num_traits::{One, Zero};
use stwo::core::channel::Blake2sM31Channel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::backend::metal::MetalBackend;
use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::TraceLocationAllocator;
use stwo_examples::wide_fibonacci::{generate_trace, FibInput, WideFibonacciComponent, WideFibonacciEval};

const FIB_SEQUENCE_LENGTH: usize = 100;

fn generate_test_trace(log_n_instances: u32) -> stwo::core::ColumnVec<CircleEvaluation<stwo::prover::backend::simd::SimdBackend, BaseField, BitReversedOrder>> {
    let inputs = (0..(1 << (log_n_instances.max(LOG_N_LANES) - LOG_N_LANES)))
        .map(|i| FibInput {
            a: PackedBaseField::one(),
            b: PackedBaseField::from_array(std::array::from_fn(|j| {
                BaseField::from_u32_unchecked((i * 16 + j) as u32)
            })),
        })
        .collect_vec();
    generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs)
}

fn main() {
    // Initialize profiling
    stwo::prover::backend::metal::profiling::init_profiling();

    let log_n_instances = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(17);

    eprintln!("\n=== Profiling Metal Proof Generation ===");
    eprintln!("log_n_instances: {}", log_n_instances);
    eprintln!("Trace size: ~{}K rows\n", (1 << log_n_instances) / 1024);

    let start = std::time::Instant::now();

    let config = PcsConfig::default();

    // Precompute twiddles
    let twiddles_start = std::time::Instant::now();
    let twiddles = MetalBackend::precompute_twiddles(
        CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    eprintln!("[TIMING] Twiddle precomputation: {:.3}ms", twiddles_start.elapsed().as_secs_f64() * 1000.0);

    // Setup protocol
    let prover_channel = &mut Blake2sM31Channel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<MetalBackend, Blake2sM31MerkleChannel>::new(
            config, &twiddles,
        );

    // Preprocessed trace (empty)
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([]);
    tree_builder.commit(prover_channel);

    // Generate trace using SIMD, then convert to Metal
    let trace_gen_start = std::time::Instant::now();
    let simd_trace = generate_test_trace(log_n_instances);
    eprintln!("[TIMING] SIMD trace generation: {:.3}ms", trace_gen_start.elapsed().as_secs_f64() * 1000.0);

    let conversion_start = std::time::Instant::now();
    let metal_trace: Vec<CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>> =
        simd_trace.iter().map(|simd_eval| {
            let cpu_values = simd_eval.values.to_cpu();
            let metal_col: Col<MetalBackend, BaseField> = cpu_values.into_iter().collect();
            CircleEvaluation::<MetalBackend, _, BitReversedOrder>::new(simd_eval.domain, metal_col)
        }).collect();
    eprintln!("[TIMING] SIMD→Metal conversion: {:.3}ms", conversion_start.elapsed().as_secs_f64() * 1000.0);

    let commit_start = std::time::Instant::now();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(metal_trace);
    tree_builder.commit(prover_channel);
    eprintln!("[TIMING] Trace commitment (FFT + Merkle): {:.3}ms", commit_start.elapsed().as_secs_f64() * 1000.0);

    // Prove constraints
    let prove_start = std::time::Instant::now();
    let component = WideFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
            log_n_rows: log_n_instances,
        },
        SecureField::zero(),
    );

    let _proof = prove::<MetalBackend, Blake2sM31MerkleChannel>(
        &[&component],
        prover_channel,
        commitment_scheme,
    )
    .expect("Metal proof generation failed");

    eprintln!("[TIMING] Proving (quotients + FRI): {:.3}ms", prove_start.elapsed().as_secs_f64() * 1000.0);

    let total = start.elapsed();
    eprintln!("\n=== Total Time: {:.3}ms ===\n", total.as_secs_f64() * 1000.0);
}
