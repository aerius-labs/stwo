//! Test GPU-resident channel state for Metal backend
//! Run with: cargo run --release --features metal_prover --example test_gpu_channel

use itertools::Itertools;
use num_traits::{One, Zero};
// GPU channel import
// use stwo::core::channel::Blake2sM31Channel;  // CPU channel (not used)
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::metal::{MetalBackend, MetalBlake2sM31Channel, MetalBlake2sM31MerkleChannel};
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
    let log_n_instances = 12; // Small test size

    eprintln!("\n=== Testing GPU-Resident Channel State ===");
    eprintln!("log_n_instances: {}", log_n_instances);
    eprintln!("Using MetalBlake2sM31MerkleChannel with GPU kernels\n");

    let start = std::time::Instant::now();

    let config = PcsConfig::default();

    // Precompute twiddles
    let twiddles = MetalBackend::precompute_twiddles(
        CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol with GPU channel
    let prover_channel = &mut MetalBlake2sM31Channel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<MetalBackend, MetalBlake2sM31MerkleChannel>::new(
            config, &twiddles,
        );

    // Preprocessed trace (empty)
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([]);
    tree_builder.commit(prover_channel);

    // Generate trace using SIMD, then convert to Metal
    let simd_trace = generate_test_trace(log_n_instances);
    let metal_trace: Vec<CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>> =
        simd_trace.iter().map(|simd_eval| {
            let cpu_values = simd_eval.values.to_cpu();
            let metal_col: Col<MetalBackend, BaseField> = cpu_values.into_iter().collect();
            CircleEvaluation::<MetalBackend, _, BitReversedOrder>::new(simd_eval.domain, metal_col)
        }).collect();

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(metal_trace);
    tree_builder.commit(prover_channel);

    // Prove constraints
    let component = WideFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
            log_n_rows: log_n_instances,
        },
        SecureField::zero(),
    );

    let _proof = prove::<MetalBackend, MetalBlake2sM31MerkleChannel>(
        &[&component],
        prover_channel,
        commitment_scheme,
    )
    .expect("GPU channel proof generation failed");

    let total = start.elapsed();
    eprintln!("\n✓ Proof generated successfully using GPU channel!");
    eprintln!("Total time: {:.3}ms", total.as_secs_f64() * 1000.0);
    eprintln!("\n=== GPU Channel Test Passed ===\n");
}
