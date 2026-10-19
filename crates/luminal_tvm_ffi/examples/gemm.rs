//! End-to-end proof: Luminal matmul -> HostOp -> TVM FFI -> cuBLAS.
//! See ../README.md for building the external shared library.
use anyhow::{Context, Result, ensure};
use cudarc::driver::DevicePtr;
use luminal::bufferize::BufferNode;
use luminal::dtype::PlanDtype;
use luminal::prelude::*;
use luminal_cuda_lite::{CudaArena, CudaRuntime, HostBuffer, as_host_op};
use luminal_tvm_ffi::{GEMM_LABEL, gemm_registration};

fn main() -> Result<()> {
    let library = std::env::args_os()
        .nth(1)
        .context("usage: gemm /absolute/path/to/gemm.so (see crates/luminal_tvm_ffi/README.md)")?;
    let mut graph = Graph::new();
    let a = graph.tensor(vec![2, 3], DType::F32);
    let b = graph.tensor(vec![3, 2], DType::F32);
    // The add also proves that ordinary Luminal work can consume the HostOp result.
    let out = a.matmul(b) + 1.0;

    // Force the FFI route for this proof. Normal users can append the row to
    // cuda_registry() and let search compare it with the existing implementations.
    let mut registry = luminal_cuda_lite::cuda_registry_without_cublaslt();
    registry.retain(|row| row.label() != "ReduceSumGeneric");
    registry.push(gemm_registration(library));
    let mut runtime = CudaRuntime::load_with_registry(&graph, registry)?;
    let mut data: FxHashMap<NodeIndex, HostBuffer> = [
        (a.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
        (b.id, vec![7.0f32, 8., 9., 10., 11., 12.].into()),
    ]
    .into_iter()
    .collect();
    let options = luminal_cuda_lite::search::CompileOptions {
        trials: 1,
        search_log: false,
        ..Default::default()
    };
    runtime.search(&Default::default(), &graph.dyn_map, &data, &options)?;
    let ffi_calls = runtime
        .plan()
        .context("missing plan")?
        .dag
        .node_weights()
        .filter(|node| {
            matches!(node, BufferNode::Compute { op, .. }
            if op.label() == GEMM_LABEL && as_host_op(op.as_ref()).is_some())
        })
        .count();
    ensure!(
        ffi_calls == 1,
        "expected one TVM FFI HostOp, found {ffi_calls}"
    );
    println!("Selected {GEMM_LABEL} as a HostOp");

    let stream = runtime.cuda_stream()?.clone();
    let mut storage = stream.alloc_zeros::<u8>(runtime.arena_bytes()?)?;
    let (ptr, record) = storage.device_ptr(&stream);
    drop(record);
    let mut staging = runtime.allocate_staging()?;
    let mut first_captures = None;
    for scale in [1.0f32, 2.0, -1.0] {
        data.insert(
            a.id,
            (1..=6).map(|v| v as f32 * scale).collect::<Vec<_>>().into(),
        );
        for (id, buffer) in &data {
            let range = runtime.input_arena_range(*id)?;
            ensure!(
                range.bytes == buffer.bytes.len(),
                "input byte count mismatch"
            );
            stream.memcpy_htod(
                &buffer.bytes,
                &mut storage.slice_mut(range.offset..range.offset + range.bytes),
            )?;
        }
        // SAFETY: storage owns the entire allocation through this synchronized execution.
        let arena = unsafe { CudaArena::from_raw(ptr, storage.len()) };
        runtime.execute(arena, &mut staging)?;
        let range = runtime.output_arena_range(out.id)?;
        let bytes = stream.clone_dtoh(&storage.slice(range.offset..range.offset + range.bytes))?;
        let backing = HostBuffer::new(PlanDtype::F32, bytes)?.as_f32()?;
        let actual = luminal_cuda_lite::layouts::dense_f32(
            &backing,
            &runtime.output_layout(out.id)?.layout,
        )?;
        let expected: Vec<_> = [58.0f32, 64., 139., 154.].map(|v| v * scale + 1.0).into();
        ensure!(actual.len() == expected.len(), "output length mismatch");
        for (got, want) in actual.iter().zip(&expected) {
            ensure!(
                (got - want).abs() <= 1e-5f32.max(want.abs() * 1e-5),
                "wrong result: expected {expected:?}, got {actual:?}"
            );
        }
        let captures = runtime
            .graph_stats()
            .context("missing graph statistics")?
            .host_captures;
        if let Some(first) = first_captures {
            ensure!(
                captures == first,
                "changing input values unexpectedly recaptured the HostOp"
            );
        } else {
            ensure!(captures > 0, "HostOp was never captured");
            first_captures = Some(captures);
        }
        println!("A scale={scale}: {actual:?} (correct)");
    }
    println!("PASS: Luminal -> TVM FFI -> cuBLAS, with CUDA graph replay");
    Ok(())
}
