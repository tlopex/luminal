#![cfg(feature = "tvm-ffi")]

mod support;

use half::bf16;
use luminal::bufferize::BufferNode;
use luminal::dtype::{DType, PlanDtype};
use luminal::graph::Graph;
use luminal::layout_ir::Sharing;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::ops::flashinfer;
use luminal_cuda_lite::{
    CudaRuntime, HostBuffer, as_host_op, cuda_registry, cuda_registry_without_cublaslt,
    harness_search_options,
};
use support::TestTransfers;

#[test]
fn matcher_requires_bf16_inputs_and_explicit_fp32_arithmetic() {
    for (dtype, cast_inputs, expected) in [
        (DType::Bf16, true, true),
        (DType::F32, true, false),
        (DType::Bf16, false, false),
    ] {
        let mut graph = Graph::new();
        let a = graph.tensor(vec![4, 16], dtype);
        let b = graph.tensor(vec![8, 16], dtype);
        let _out = if cast_inputs {
            a.cast(DType::F32)
                .matmul(b.cast(DType::F32).permute(&[1, 0]))
        } else {
            a.matmul(b.permute(&[1, 0])).cast(DType::F32)
        };
        // Both libraries coexist in the registry; saturation never loads a DLL.
        let mut registry = cuda_registry();
        registry.push(flashinfer::gemm("unused-until-preparation.so", 0));
        let runtime = CudaRuntime::load_with_registry(&graph, registry).unwrap();
        let egraph = runtime.saturated_egraph(&Default::default()).unwrap();
        let admitted = egraph
            .nodes
            .values()
            .any(|node| node.op == "LayoutTensorOpFlashInferGemm");
        if expected {
            assert!(
                egraph
                    .nodes
                    .values()
                    .any(|node| node.op == "LayoutTensorOpCublasLt")
            );
        }
        assert_eq!(
            admitted, expected,
            "dtype={dtype:?}, cast_inputs={cast_inputs}"
        );
    }
}

#[test]
fn dps_writes_only_its_destination_and_reserves_the_configured_scratch() {
    let entry = flashinfer::gemm("provider.so", 123456);
    let dps = entry.prototype.to_dps().unwrap();
    assert!(dps.operand_reads_memory(0) && dps.operand_reads_memory(1));
    assert!(!dps.operand_reads_memory(2));
    let aliases = dps.alias_info();
    assert_eq!(aliases.len(), 1);
    assert_eq!((aliases[0].operand, aliases[0].result), (2, 0));
    assert!(matches!(aliases[0].sharing, Sharing::Must));
    let cloned = dps.clone();
    assert_eq!(
        as_host_op(cloned.as_ref())
            .unwrap()
            .workspace_bytes(&Default::default())
            .unwrap(),
        123456
    );
}

fn payload(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| bf16::from_f32(((i * 7 + seed * 11) % 29) as f32 / 13.0 - 1.0).to_f32())
        .collect()
}

fn bf16_buffer(values: &[f32]) -> HostBuffer {
    HostBuffer::new(
        PlanDtype::Bf16,
        values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_bits().to_ne_bytes())
            .collect(),
    )
    .unwrap()
}

/// Run with --ignored and FLASHINFER_GEMM_LIBRARY pointing to the upstream
/// v0.7.2 mm_bf16_cublaslt TVM FFI module or its JSON build recipe.
#[test]
#[ignore = "requires the FlashInfer GEMM TVM FFI library; set FLASHINFER_GEMM_LIBRARY"]
fn upstream_gemm_is_elected_captured_and_replayed_with_new_inputs() {
    let library = std::env::var("FLASHINFER_GEMM_LIBRARY").expect("set FLASHINFER_GEMM_LIBRARY");
    for (m, n, k) in [(4, 8, 16), (16, 24, 32)] {
        let mut graph = Graph::new();
        let a = graph.tensor(vec![m, k], DType::Bf16);
        let b = graph.tensor(vec![n, k], DType::Bf16);
        let out = a
            .cast(DType::F32)
            .matmul(b.cast(DType::F32).permute(&[1, 0]));
        // Require this provider in the execution test. The matcher test above
        // keeps the full registry, including cuBLASLt, to check coexistence.
        let mut registry = cuda_registry_without_cublaslt();
        registry.retain(|row| row.label() != "ReduceSumGeneric");
        registry.push(flashinfer::gemm(&library, 32 * 1024 * 1024));
        let mut runtime = CudaRuntime::load_with_registry(&graph, registry).unwrap();
        let a_values = payload(m * k, 1);
        let b_values = payload(n * k, 2);
        let data: FxHashMap<_, _> = [
            (a.id, bf16_buffer(&a_values)),
            (b.id, bf16_buffer(&b_values)),
        ]
        .into_iter()
        .collect();
        let options = luminal_cuda_lite::CompileOptions {
            generations: 12,
            generation_size: 16,
            mutations: 4,
            seed: 0,
            ..harness_search_options()
        };
        runtime
            .search(&Default::default(), &Default::default(), &data, &options)
            .unwrap();
        assert!(runtime.plan().unwrap().dag.node_weights().any(
            |node| matches!(node, BufferNode::Compute { op, .. } if op.label() == "FlashInferGemm")
        ));
        let mut arena = support::allocate_arena(&runtime).unwrap();
        let mut staging = runtime.allocate_staging().unwrap();
        for seed in [1, 3] {
            let a_values = payload(m * k, seed);
            let b_values = payload(n * k, seed + 1);
            let a_data = bf16_buffer(&a_values);
            let b_data = bf16_buffer(&b_values);
            runtime.upload(&mut arena, a.id, a_data.clone()).unwrap();
            runtime.upload(&mut arena, b.id, b_data.clone()).unwrap();
            runtime.execute(arena.arena(), &mut staging).unwrap();
            let got = runtime.read_f32(&arena, out.id).unwrap();
            for row in 0..m {
                for col in 0..n {
                    let expected: f32 = (0..k)
                        .map(|i| a_values[row * k + i] * b_values[col * k + i])
                        .sum();
                    let actual = got[row * n + col];
                    assert!(
                        (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                        "({m},{n},{k})[{row},{col}]: {actual} != {expected}"
                    );
                }
            }
            // The provider's contract promises both inputs remain read-only.
            let a_home = runtime.input_arena_range(a.id).unwrap();
            let b_home = runtime.input_arena_range(b.id).unwrap();
            assert_eq!(
                arena.read(a_home.offset, a_home.bytes).unwrap(),
                a_data.bytes
            );
            assert_eq!(
                arena.read(b_home.offset, b_home.bytes).unwrap(),
                b_data.bytes
            );
        }
    }
}
