# Rust / TVM FFI GEMM prototype

This standalone example validates the call path used by the Luminal TVM FFI
HostOp:

```text
Rust GPU buffers
  -> TVM FFI Module::load_from_file / get_function
  -> Function::call_packed
  -> gemm.so: __tvm_ffi_gemm
  -> cuBLAS SGEMM on the caller's CUDA stream
```

It has its own Cargo workspace and does not change Luminal's registry, search,
DPS pass, or bufferizer. The integrated HostOp lives in
[`crates/luminal_tvm_ffi`](../../crates/luminal_tvm_ffi/README.md).

## Run

Requirements: Linux, an NVIDIA GPU, a matching CUDA toolkit including cuBLAS,
a C++17 compiler, Rust 1.91 or newer, and `apache-tvm-ffi==0.1.12` in the active
Python environment. `tvm-ffi-config` and `cargo` must be on `PATH`.

The Rust TVM FFI dependency is pinned to the commit tagged `v0.1.12`, and cudarc
is pinned to the same commit as Luminal. The installed TVM FFI package supplies
the C++ headers and shared runtime. Python is used for build configuration;
the resulting Rust executable invokes the native function without a Python
interpreter in the execution path.

From the repository root, selecting a CUDA toolkit that supports your GPU:

```bash
CUDA_ROOT=/usr/local/cuda-13.1 CUDA_VISIBLE_DEVICES=0 \
  examples/tvm_ffi_gemm/run.sh
```

The script builds `target/gemm.so` inside this example directory, sets library
search paths for the command, and builds/runs the Rust program. The small case
prints:

```text
PASS eager 2x3 @ 3x2: [58.0, 64.0, 139.0, 154.0]; inputs unchanged
```

## Interface and contract

`gemm(handle_address, A, B, out, workspace)` is exported by
`TVM_FFI_DLL_EXPORT_TYPED_FUNC` in `gemm.cc`.

- `handle_address` is a live cuBLAS handle created outside graph capture, on the
  tensors' device. It is borrowed and exclusively used during a call. This
  wrapper sets its stream, workspace, pointer mode, and math mode.
- `A[M,K]`, `B[K,N]`, and `out[M,N]` are contiguous, row-major FP32 CUDA tensors
  with zero byte offsets. Dimensions must be positive and fit the cuBLAS `int`
  API. All tensor allocations must be large enough for their descriptors.
- The operation computes `out = A @ B`. `A` and `B` are read-only. Every output
  element is written; the old output is not read (`beta = 0`). Output/input
  overlap is rejected. The two read-only inputs may overlap each other.
- `workspace` is a contiguous CUDA uint8 tensor of at least 32 MiB, aligned to
  256 bytes and disjoint from A, B, and out. The caller owns all tensor and
  workspace allocations. The wrapper does not allocate tensor storage.
- The CUDA stream is borrowed from TVM FFI's per-device thread-local stream
  context. Rust sets and restores that context around the call, including when
  the function returns an error.
- The handle, module, and GPU allocations must remain valid until asynchronous
  work and all graph replays finish. Input updates and output reads must be
  ordered on the same stream or explicitly synchronized.

## Checks performed by the executable

- Four rectangular/degenerate matrix shapes, compared to an independent CPU
  reference; includes both single-row and single-reduction-dimension cases.
- Outputs initially contain NaNs, checking the write-only destination contract.
- A and B remain unchanged after eager calls and graph replays.
- CUDA Graph capture on a non-default stream, starting before the first GEMM
  call; capture itself leaves the output unchanged.
- Two replays per shape with updated inputs at the same addresses.
- Wrong dtype, incompatible shapes, and output/input aliasing return FFI errors.
- The original TVM FFI stream context is restored after successful and failed
  calls.

Verified on an RTX 5090 with CUDA 13.1, Rust 1.97.1, and TVM FFI 0.1.12.
The executable also passed CUDA Compute Sanitizer memcheck with `0 errors`,
and `cargo clippy --all-targets --no-deps -- -D warnings` passed for this crate.

## Luminal integration

[`luminal_tvm_ffi`](../../crates/luminal_tvm_ffi/README.md) registers this library
as a real GEMM search candidate, supplies output and workspace from Luminal's
arena, and declares matching semantics, DPS, memory effects, and aliasing. These
declarations are not inferred from the FFI function. Its `gemm` example verifies
the complete graph -> HostOp -> TVM FFI -> cuBLAS -> downstream add path.
