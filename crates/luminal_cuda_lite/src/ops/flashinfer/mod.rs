//! FlashInfer BF16 GEMM through its existing TVM FFI exports.
//!
//! Computes `A.cast(F32) @ B.cast(F32).transpose()` for dense row-major
//! `A[m,k]`, `B[n,k]`, and `D[m,n]`. Inputs are read-only; DPS supplies D,
//! and all GPU scratch comes from the runtime arena.
//! Matching requires positive literal matrix extents fitting int32 geometry.
//!
//! Append `flashinfer::gemm(library_or_recipe, scratch_bytes)` to the caller's
//! registry. Execution requires the `tvm-ffi` feature. A `.json` recipe is built
//! automatically during preparation through the shared TVM FFI SDK builder;
//! a prebuilt library is loaded directly. The exported ABI is FlashInfer v0.7.2's
//! [`csrc/mm_bf16_cublaslt.cu`](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/csrc/mm_bf16_cublaslt.cu)
//! (the cuBLASLt backend of `flashinfer.gemm.mm_bf16`). To use this directory's
//! `build.json`, set `FLASHINFER_SRC` to that checkout and `CUDA_ROOT` to the
//! toolkit root. The SDK locates nvcc via `CUDA_HOME`/`CUDA_PATH` or `PATH`.

mod device_call;

use crate::host::TvmFfiLibrary;
use std::path::PathBuf;

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::{CudaOpInterface, HostOp, RegisteredOp};

/// Add a GEMM candidate using a prebuilt library or automatic JSON build recipe.
/// `scratch_bytes` is the maximum workspace the algorithm query may select.
pub fn gemm(library: impl Into<PathBuf>, scratch_bytes: usize) -> RegisteredOp {
    let op = FlashInferGemm {
        library: TvmFfiLibrary::new(library),
        scratch_bytes,
        dps: false,
    };
    RegisteredOp::new(Box::new(GemmMatcher(op.clone())), Box::new(op))
}

#[derive(Debug, Clone)]
struct FlashInferGemm {
    library: TvmFfiLibrary,
    scratch_bytes: usize,
    dps: bool,
}

impl OpSlotNames for FlashInferGemm {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "a".into(),
            1 => "b".into(),
            _ => "dest0".into(),
        }
    }
}

impl BufferTensorIrOp for FlashInferGemm {
    fn label(&self) -> &str {
        "FlashInferGemm"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        !self.dps || operand < 2
    }

    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        self.dps
            .then(|| CudaOpInterface::host::<Self>() as &dyn std::any::Any)
    }
}

impl Bufferizable for FlashInferGemm {
    fn alias_info(&self) -> Vec<AliasInfo> {
        if self.dps {
            vec![AliasInfo {
                operand: 2,
                result: 0,
                sharing: Sharing::Must,
            }]
        } else {
            vec![]
        }
    }
}

impl ToDps for FlashInferGemm {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        (!self.dps).then(|| {
            Box::new(Self {
                dps: true,
                ..self.clone()
            }) as Box<dyn LayoutIrOp>
        })
    }
}

impl LayoutIrOp for FlashInferGemm {}

impl HostOp for FlashInferGemm {
    fn workspace_bytes(&self, _bounds: &crate::symbolic::Bounds) -> anyhow::Result<usize> {
        Ok(self.scratch_bytes)
    }

    unsafe fn prepare(
        &self,
        ctx: &crate::host::HostOpContext<'_>,
    ) -> anyhow::Result<Box<dyn crate::host::PreparedHostOp>> {
        // SAFETY: forwarded from HostOp::prepare's allocation/alias contract.
        unsafe { device_call::prepare(&self.library, ctx) }
    }
}

#[derive(Debug)]
struct GemmMatcher(FlashInferGemm);

impl OpMatcher for GemmMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpFlashInferGemm"
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 2)]
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LayoutOpConstructors,
                text: "(constructor LayoutTensorOpFlashInferGemm (LayoutTensor LayoutTensor Layout) LayoutTensorOp)",
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("egg/match_gemm.egg"),
            },
        ]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(self.0.clone())
    }
}
