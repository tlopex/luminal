//! Optional external-library operations, selected by egglog and executed as CUDA HostOps.
//!
//! Add [`gemm_registration`] to a CUDA runtime registry to offer an FP32 GEMM
//! exported through TVM FFI. The `device` feature enables native FFI execution;
//! graph matching, DPS, and alias analysis remain usable without TVM FFI installed.

mod gemm;
#[cfg(feature = "device")]
mod native;

pub use gemm::{GEMM_CONSTRUCTOR, GEMM_LABEL, TvmFfiGemm, gemm_registration};

/// Scratch is explicitly supplied by the caller, never allocated by the GEMM wrapper.
pub const GEMM_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
