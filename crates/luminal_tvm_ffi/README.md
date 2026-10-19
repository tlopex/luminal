# Minimal TVM FFI GEMM HostOp

One path: `Luminal matmul -> HostOp -> TVM FFI -> cuBLAS SGEMM`.
This PoC uses SGEMM; Luminal's existing cuBLASLt adapter is separate.

| File | Purpose |
| --- | --- |
| `src/lib.rs` | Registration, DPS, alias/mutation declarations, contract tests |
| `src/match.egg` | Matrix multiplication semantics and supported layouts |
| `src/native.rs` | Load the FFI module, describe tensors, record the packed call |
| `gemm.cc` | Export `gemm(handle, A, B, out, workspace)` and call cuBLAS |
| `examples/gemm.rs` | Verify HostOp selection and one GPU result for `A @ B + 1` |
| `run.sh` | Build the shared library and run that example |

The contract is contiguous row-major FP32 `A[M,K] @ B[K,N]`, with positive
int32 dimensions. A/B are read-only; the output is written without reading
its previous contents and must not overlap A/B. DPS ties the result to the
destination. Luminal supplies 32 MiB of aligned, disjoint scratch storage.
The native bridge prepares resources outside capture and sets/restores TVM
FFI's stream context around recording. Resources survive all graph replays.

## Run

Requires Linux, an NVIDIA GPU, a compatible CUDA toolkit with cuBLAS, a C++17
compiler, Rust 1.91+, and `apache-tvm-ffi==0.1.12` in the active Python environment.
`cargo` and `tvm-ffi-config` must be on PATH. From the repository root:

```bash
CUDA_ROOT=/usr/local/cuda-13.1 CUDA_VISIBLE_DEVICES=0 crates/luminal_tvm_ffi/run.sh
```

The script builds `target/tvm_ffi_gemm.so` and runs the single integrated example.
That example forces the FFI route in its own registry so it actually tests this
HostOp. Ordinary integration appends `gemm_registration("/path/to/library.so")`
to the chosen CUDA registry. The exported signature is specific to this wrapper.

The four analysis tests need no GPU or installed TVM FFI runtime:

```bash
cargo test -p luminal_tvm_ffi
```
