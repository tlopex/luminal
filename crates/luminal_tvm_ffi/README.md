# TVM FFI GEMM HostOp proof of concept

This optional crate adds one external GEMM to Luminal's CUDA search space:

```text
a.matmul(b) + 1
  -> egglog selects TvmFfiGemm for the matmul
  -> DPS supplies the output buffer; bufferization enforces aliasing
  -> HostOp::prepare loads gemm.so and creates a cuBLAS handle
  -> PreparedHostOp::record calls TVM FFI inside CUDA graph capture
  -> gemm.so calls cuBLAS on Luminal's stream
  -> graph replay computes GEMM, then Luminal's add consumes the result
```

The first implementation supports positive, contiguous row-major FP32 matrices
`A[M,K] @ B[K,N]`. Dimensions must fit the cuBLAS int32 API. The restrictions
are checked by the egglog matcher and validated again when preparing the call.
It does not yet provide batched GEMM, other dtypes, or a generic registration
schema for arbitrary third-party FFI functions.

## Run the complete example

Requirements: Linux, an NVIDIA GPU, a compatible CUDA toolkit with cuBLAS, a
C++17 compiler, Rust 1.91+, and `apache-tvm-ffi==0.1.12` installed in the active
Python environment (`pip install apache-tvm-ffi==0.1.12`). `cargo` and
`tvm-ffi-config` must be on PATH. The Rust FFI dependency is pinned to the
matching upstream release commit.

From the repository root (adjust the CUDA toolkit path for your machine):

```bash
export CUDA_ROOT=/usr/local/cuda-13.1
export CUDA_VISIBLE_DEVICES=0
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$(tvm-ffi-config --libdir):$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Build gemm.so and validate the independent Rust -> FFI -> cuBLAS bridge.
examples/tvm_ffi_gemm/run.sh

# Run the same library as a real Luminal HostOp.
cargo run -p luminal_tvm_ffi --features device --example gemm -- \
  "$PWD/examples/tvm_ffi_gemm/target/gemm.so"
```

The example checks that the plan contains one `TvmFfiGemm` HostOp, verifies
`[[59, 65], [140, 155]]` for `A @ B + 1`, and replays the same graph with two
updated inputs without recapturing the HostOp. It removes the existing cuBLASLt
and decomposed reduction routes from **this example's registry** to ensure the
new path is exercised. It makes no performance comparison.

Verified on an RTX 5090 with CUDA 13.1 and TVM FFI 0.1.12. The integrated
example also passed CUDA Compute Sanitizer memcheck with zero errors.

Normal integration simply appends the registration to a registry:

```rust,ignore
let mut registry = luminal_cuda_lite::cuda_registry();
registry.push(luminal_tvm_ffi::gemm_registration("/path/to/gemm.so"));
let runtime = luminal_cuda_lite::CudaRuntime::load_with_registry(&graph, registry)?;
```

## What Luminal must know

TVM FFI supplies the calling convention. These compiler declarations are still
explicit:

| Requirement | Implementation |
| --- | --- |
| Semantics and supported layouts | `src/match.egg`: recognizes matrix multiplication and requires rank-2 FP32 contiguous layouts |
| DPS | `ToDps`: changes `(A, B) -> out` into `(A, B, dest) -> out` |
| Mutation and aliasing | `Bufferizable`: result must share `dest`; result must not overlap A/B; A/B are read-only; old dest is not read |
| Scratch allocation | `HostOp::workspace_bytes`: 32 MiB reserved in Luminal's arena, aligned by the runtime |
| Execution | `src/native.rs`: loads the module once per registration, prepares descriptors and a handle per call, sets/restores TVM FFI's stream context, and calls the packed function during capture |

The FFI signature is `gemm(handle_address, A, B, out, workspace)`, exported with
`TVM_FFI_DLL_EXPORT_TYPED_FUNC`. The handle argument is specific to this small
wrapper; it is not a universal GEMM signature imposed by TVM FFI. The
[external library's contract](../../examples/tvm_ffi_gemm/README.md) describes
tensor lifetimes, layout, and workspace ownership.

Analysis-only checks need neither a GPU nor an installed TVM FFI runtime:

```bash
cargo test -p luminal_tvm_ffi
```

Enabling `device` links TVM FFI and enables execution. Ordinary Luminal CUDA
users do not acquire a TVM FFI runtime dependency unless they enable this crate's
device feature.
