#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${CUDA_ROOT:?Set CUDA_ROOT to a toolkit supporting your GPU, e.g. /usr/local/cuda-13.1}"

export PATH="$CUDA_ROOT/bin:$PATH"
ffi_libdir="$(tvm-ffi-config --libdir)"
export LD_LIBRARY_PATH="$ffi_libdir:$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$example_dir/target}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
mkdir -p "$CARGO_TARGET_DIR"

# tvm-ffi-config emits compiler flags; arrays preserve each flag as one argument.
read -r -a ffi_cxxflags <<< "$(tvm-ffi-config --cxxflags)"
read -r -a ffi_ldflags <<< "$(tvm-ffi-config --ldflags)"
read -r -a ffi_libs <<< "$(tvm-ffi-config --libs)"

"${CXX:-c++}" -std=c++17 -O2 -shared -fPIC -fvisibility=hidden \
  "${ffi_cxxflags[@]}" -I"$CUDA_ROOT/include" \
  "$example_dir/gemm.cc" -o "$CARGO_TARGET_DIR/gemm.so" \
  "${ffi_ldflags[@]}" "${ffi_libs[@]}" \
  -L"$CUDA_ROOT/lib64" -lcublas -lcudart \
  -Wl,-rpath,"$ffi_libdir" -Wl,-rpath,"$CUDA_ROOT/lib64"

cargo run --manifest-path "$example_dir/Cargo.toml" -- "$CARGO_TARGET_DIR/gemm.so"
