// TVM FFI export called by Luminal's GEMM HostOp.
#include <cublas_v2.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <limits>

namespace {
using tvm::ffi::TensorView;
constexpr int64_t kWorkspaceBytes = 32 * 1024 * 1024;

void CheckBlas(cublasStatus_t status, const char *operation) {
  TVM_FFI_CHECK(status == CUBLAS_STATUS_SUCCESS, RuntimeError)
      << operation << " failed with cuBLAS status " << static_cast<int>(status);
}

void CheckTensor(TensorView tensor, int rank, DLDataType dtype, const char *name) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.ndim() == rank, ValueError) << name << " has the wrong rank";
  TVM_FFI_CHECK(tensor.dtype() == dtype, TypeError) << name << " has the wrong dtype";
  TVM_FFI_CHECK(tensor.byte_offset() == 0, ValueError) << name << " must have zero byte_offset";
  TVM_FFI_CHECK(tensor.IsContiguous(), ValueError) << name << " must be contiguous";
  TVM_FFI_CHECK(tensor.data_ptr() != nullptr, ValueError) << name << " has a null data pointer";
  for (int i = 0; i < rank; ++i) {
    TVM_FFI_CHECK(tensor.size(i) > 0, ValueError) << name << " must have positive dimensions";
    if (rank == 2) {
      TVM_FFI_CHECK(tensor.size(i) <= std::numeric_limits<int>::max(), ValueError)
          << name << " dimensions must fit cuBLAS's int interface";
    }
  }
}

bool Overlaps(TensorView a, uint64_t a_bytes, TensorView b, uint64_t b_bytes) {
  auto a_ptr = reinterpret_cast<uintptr_t>(a.data_ptr());
  auto b_ptr = reinterpret_cast<uintptr_t>(b.data_ptr());
  // Subtraction avoids overflowing an end address.
  return a_ptr <= b_ptr ? b_ptr - a_ptr < a_bytes : a_ptr - b_ptr < b_bytes;
}

void Gemm(int64_t handle_address, TensorView a, TensorView b, TensorView out,
          TensorView workspace) {
  constexpr DLDataType f32{kDLFloat, 32, 1};
  constexpr DLDataType u8{kDLUInt, 8, 1};
  CheckTensor(a, 2, f32, "A");
  CheckTensor(b, 2, f32, "B");
  CheckTensor(out, 2, f32, "out");
  CheckTensor(workspace, 1, u8, "workspace");
  for (auto tensor : {b, out, workspace}) {
    TVM_FFI_CHECK(tensor.device().device_id == a.device().device_id, ValueError)
        << "all tensors must be on the same CUDA device";
  }
  TVM_FFI_CHECK(a.size(1) == b.size(0), ValueError) << "incompatible GEMM shapes";
  TVM_FFI_CHECK(out.size(0) == a.size(0) && out.size(1) == b.size(1), ValueError)
      << "out has the wrong shape";
  TVM_FFI_CHECK(handle_address != 0, ValueError) << "cuBLAS handle must be non-null";
  TVM_FFI_CHECK(workspace.size(0) >= kWorkspaceBytes, ValueError)
      << "workspace must contain at least " << kWorkspaceBytes << " bytes";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 256 == 0, ValueError)
      << "workspace must be aligned to 256 bytes";

  auto a_bytes = static_cast<uint64_t>(a.numel()) * sizeof(float);
  auto b_bytes = static_cast<uint64_t>(b.numel()) * sizeof(float);
  auto out_bytes = static_cast<uint64_t>(out.numel()) * sizeof(float);
  auto workspace_bytes = static_cast<uint64_t>(workspace.size(0));
  TVM_FFI_CHECK(!Overlaps(a, a_bytes, out, out_bytes) && !Overlaps(b, b_bytes, out, out_bytes),
                ValueError)
      << "out must not alias A or B";
  TVM_FFI_CHECK(!Overlaps(a, a_bytes, workspace, workspace_bytes) &&
                    !Overlaps(b, b_bytes, workspace, workspace_bytes) &&
                    !Overlaps(out, out_bytes, workspace, workspace_bytes),
                ValueError)
      << "workspace must not alias A, B, or out";

  tvm::ffi::CUDADeviceGuard device_guard(a.device().device_id);
  auto stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(a.device().device_type, a.device().device_id));
  auto handle = reinterpret_cast<cublasHandle_t>(static_cast<uintptr_t>(handle_address));
  // cublasSetStream resets the workspace, so bind the caller's workspace
  // afterwards.
  CheckBlas(cublasSetStream(handle, stream), "cublasSetStream");
  CheckBlas(cublasSetWorkspace(handle, workspace.data_ptr(), workspace_bytes),
            "cublasSetWorkspace");
  CheckBlas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
  CheckBlas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
  const float alpha = 1.0f;
  const float beta = 0.0f;  // Write-only destination: the old output is not an accumulator.
  int m = static_cast<int>(a.size(0));
  int k = static_cast<int>(a.size(1));
  int n = static_cast<int>(b.size(1));
  // cuBLAS is column-major: out^T[n,m] = B^T[n,k] * A^T[k,m].
  CheckBlas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                        static_cast<const float *>(b.data_ptr()), n,
                        static_cast<const float *>(a.data_ptr()), k, &beta,
                        static_cast<float *>(out.data_ptr()), n),
            "cublasSgemm");
}
}  // namespace

// Generates the ABI symbol __tvm_ffi_gemm.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm, Gemm);
