// Only an ABI adapter. Descriptors, algorithm, buffers, and scratch are prepared
// by device_call.rs using Luminal's existing cuBLASLt HostOp contracts.
// Set LUMINAL_CUBLASLT_TVM_FFI_LIBRARY to build.json for automatic compilation,
// or to an existing shared library. The recipe requires CUDA_ROOT set to the
// toolkit root and the cublaslt-tvm-ffi feature.
#include <cublasLt.h>
#include <tvm/ffi/function.h>

#include <cstdint>

namespace {
int64_t Matmul(int64_t handle, int64_t desc, int64_t alpha, int64_t beta, int64_t a,
               int64_t a_layout, int64_t b, int64_t b_layout, int64_t c, int64_t c_layout,
               int64_t d, int64_t d_layout, int64_t algo, int64_t workspace,
               int64_t workspace_bytes, int64_t stream) {
  return static_cast<int64_t>(cublasLtMatmul(
      reinterpret_cast<cublasLtHandle_t>(handle), reinterpret_cast<cublasLtMatmulDesc_t>(desc),
      reinterpret_cast<const void*>(alpha), reinterpret_cast<const void*>(a),
      reinterpret_cast<cublasLtMatrixLayout_t>(a_layout), reinterpret_cast<const void*>(b),
      reinterpret_cast<cublasLtMatrixLayout_t>(b_layout), reinterpret_cast<const void*>(beta),
      reinterpret_cast<const void*>(c), reinterpret_cast<cublasLtMatrixLayout_t>(c_layout),
      reinterpret_cast<void*>(d), reinterpret_cast<cublasLtMatrixLayout_t>(d_layout),
      reinterpret_cast<const cublasLtMatmulAlgo_t*>(algo), reinterpret_cast<void*>(workspace),
      static_cast<size_t>(workspace_bytes), reinterpret_cast<cudaStream_t>(stream)));
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_matmul, Matmul);
