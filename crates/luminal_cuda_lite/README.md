# CUDA Lite execution

CUDA Lite has one execution path: a CUDA graph launch. Serving, candidate warmup,
and profiling use `CudaDevice::execute`. Graph compilation and instantiation are
outside timed trials. An execution stages inputs, launches the graph, waits for
completion, and returns owned host output bytes.

## Compilation and memory

Egglog selects all operations. Bucket searches extract from range-seeded e-graphs,
so implementation guards must hold throughout the interval. Representative
shapes are used for timing, without freezing the installed plan's geometry.

The storage planner evaluates conservative interval capacities for each buffer.
It respects the bufferizer's allocation/free events and data/anti-dependencies.
One schedule places input uploads immediately before their first device use,
then bufferized operations and output readbacks at their output boundaries.
Graph construction consumes this schedule directly.

One interval allocator packs every physical resource:

- The dynamic-dimension parameter block remains live throughout execution.
- Interior tensors follow their bufferizer alloc/free markers.
- Donated device copies end at their explicit free; other boundary copies end
  at their final device use, including any required readback.
- Escaping device outputs start at their allocation and end after their final
  use/readback. Returned host bytes keep the caller's output alive separately.
- Each HostOp's scratch is live only during that operation's child graph, and
  can reuse storage occupied by tensors or other HostOps at different times.

Pinned staging uses the same allocator with byte alignment. All input payloads
remain live from host preparation through their upload; each output remains live
from readback through host result collection. Completed uploads can therefore
provide space for outputs, while late inputs remain protected. Multiple output
slots sharing a buffer at one boundary share a single readback and staging range.

Each bucket keeps its capacity-sized offsets fixed as live dimensions change.
The arena allocation is the **maximum** requirement across installed buckets.
Pinned host staging is shared across buckets too. Execution is serialized on one
nonblocking stream; buckets cannot run concurrently against these shared ranges.
Returned output data owns its host memory and survives later launches.

Installing a new plan set synchronizes and destroys old executables before any
allocation can move. Search releases candidate graphs, staging, and arena memory
between candidates, while retaining the CUDA context and compiled module cache.

## Dynamic updates

| Change | Work before graph launch |
| --- | --- |
| Input contents | Fill the existing pinned staging ranges. |
| Dimension used only inside kernels | Write the parameter block; no graph node updates. |
| Copy length | Patch dependent copy nodes; disable zero-length copies. |
| Explicit kernel launch geometry | Patch dependent kernel nodes. |
| HostOp capture dimensions | Reuse a cached capture, or prepare and capture that HostOp. |
| Compatible child graph topology | Update the existing executable's child node. |
| Incompatible child graph topology | Rebind a cached parent executable, or instantiate a new one. |
| Bucket switch | Select that bucket's graph using the same arena and staging allocations. |

Dimension-to-node dependencies are built once. Host captures and parent topology
variants have bounded caches (eight each). Parents retain the preparations
referenced by their source and executable nodes, even after capture-cache
eviction. A failed partial update invalidates
the affected compiled plan before another launch can use it.

## Operation interfaces

`KernelOp::codegen` returns CUDA source plus symbolic launch metadata. The kernel
ABI is input pointers, output pointer, then `const long long* params`. Dimension
identifiers returned by `symbolic::variable` are defined as entries in `params`.
Default launches cover the bucket capacity and kernels guard against the live
extent. `KernelSource::launch` can instead specify symbolic grid, block, and
shared-memory geometry through `KernelLaunch`.

`HostOp::prepare` resolves host descriptors and algorithms outside capture.
`PreparedHostOp::record` submits GPU work during capture; its Rust body is not
called on replay. The executor treats the resulting child graph as opaque.
`workspace_bytes` reserves operation-local device scratch in the shared arena;
preparation must not access scratch contents or preserve them beyond that
operation's captured work. `capture_dims`
may narrow invalidation to the dimensions that affect captured work; its default
conservatively depends on all dimensions.

cuBLASLt uses this interface for all DPS forms. Geometry changes rebuild its
public library descriptors and capture a new child graph. The executor never
inspects or edits cuBLASLt's private kernel arguments.

The `tvm-ffi` feature lets CUDA HostOps call external libraries through TVM FFI.
The SDK is re-exported as `luminal_cuda_lite::tvm_ffi`. Use
`host::TvmFfiLibrary::new(config_path).load()` and resolve the function during
preparation, then call
`unsafe { capture.call_tvm_ffi(&function, &args) }` during recording. Arguments
and return values follow the library's signature; calls use the capture stream.

Install `apache-tvm-ffi==0.1.15` and make `tvm-ffi-config` available on `PATH`.
The configuration can be a prebuilt library or a `.json` recipe containing
keyword arguments for `tvm_ffi.cpp.build`. Recipes automatically build on first
preparation and use the SDK's persistent cache. Cloned operations share the
resolved artifact. Paths are relative to the recipe; strings expand environment
variables. Source builds also require Ninja and the compiler/toolkit. Select
Python with `LUMINAL_TVM_FFI_PYTHON` and the cache with `TVM_FFI_CACHE_DIR`.

The [cuBLASLt recipe](src/ops/cublaslt/build.json) can be selected with
`LUMINAL_CUBLASLT_TVM_FFI_LIBRARY` under `cublaslt-tvm-ffi`. The
[FlashInfer recipe](src/ops/flashinfer/build.json) can be passed to
`ops::flashinfer::gemm(config_path, scratch_bytes)`. Set `CUDA_ROOT` to the
toolkit root and `FLASHINFER_SRC` to FlashInfer's source checkout for its recipe;
the SDK locates nvcc using `CUDA_HOME`/`CUDA_PATH` or `PATH`. Recipes supply the
provider-specific sources and flags; the shared builder does not select libraries.

For FlashInfer GEMM, append `ops::flashinfer::gemm(config_path, scratch_bytes)`
to the registry passed to `CudaRuntime::load_with_registry`. The candidate
describes BF16 inputs with FP32 arithmetic and output; its module documents the
upstream TVM FFI exports.

During bucket profiling, dynamic payloads supplied at another size are truncated
or zero-extended to the representative size for timing. Serving requires exact
live payload sizes. Profiling includes staging and readback, matching serving.

`CudaRuntime::graph_stats` exposes counters and arena/staging footprint for
checking reuse. GPU regressions live in `tests/dynamic_graphs.rs` and
`tests/execution_traits.rs`.
