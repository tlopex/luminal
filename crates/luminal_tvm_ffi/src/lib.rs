//! Optional external-library operations, selected by egglog and executed as CUDA HostOps.
//!
//! Add [`gemm_registration`] to a CUDA runtime registry to offer an FP32 GEMM
//! exported through TVM FFI. The `device` feature enables native FFI execution;
//! graph matching, DPS, and alias analysis remain usable without TVM FFI installed.

#[cfg(feature = "device")]
mod native;

/// Scratch supplied by Luminal's arena.
pub const GEMM_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

use std::path::PathBuf;
use std::rc::Rc;

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};
use luminal_cuda_lite::{CudaOpInterface, HostOp, RegisteredOp};

pub const GEMM_CONSTRUCTOR: &str = "LayoutTensorOpTvmFfiGemm";
pub const GEMM_LABEL: &str = "TvmFfiGemm";

/// The implementation library, shared across candidate instances. Each prepared
/// call owns a separate cuBLAS handle; loading the module is cached per registration.
pub(crate) struct Library {
    pub path: PathBuf,
    #[cfg(feature = "device")]
    pub loaded: std::cell::RefCell<Option<crate::native::LoadedFunction>>,
}

impl std::fmt::Debug for Library {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GemmLibrary")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// `A[M,K], B[K,N] -> out[M,N]`, with row-major contiguous FP32 storage.
/// The matched equation and layout restrictions live in `match.egg`.
#[derive(Debug, Clone)]
pub struct TvmFfiGemm {
    pub(crate) library: Rc<Library>,
    dps: bool,
}

impl OpSlotNames for TvmFfiGemm {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "a".into(),
            1 => "b".into(),
            2 if self.dps => "dest".into(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TvmFfiGemm {
    fn label(&self) -> &str {
        GEMM_LABEL
    }

    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        self.dps
            .then(|| CudaOpInterface::host::<Self>() as &dyn std::any::Any)
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand < 2
    }
}

impl Bufferizable for TvmFfiGemm {
    fn alias_info(&self) -> Vec<AliasInfo> {
        if self.dps {
            // The result is the destination's storage. No May edges: A/B
            // cannot alias the written result. Read/read sharing is safe.
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

impl ToDps for TvmFfiGemm {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        (!self.dps).then(|| {
            Box::new(Self {
                library: self.library.clone(),
                dps: true,
            }) as _
        })
    }
}

impl LayoutIrOp for TvmFfiGemm {}

impl HostOp for TvmFfiGemm {
    fn workspace_bytes(&self, _: &luminal_cuda_lite::symbolic::Bounds) -> anyhow::Result<usize> {
        Ok(crate::GEMM_WORKSPACE_BYTES)
    }

    // Default capture_dims=None conservatively tracks every dynamic dimension.
    #[cfg(feature = "device")]
    unsafe fn prepare(
        &self,
        ctx: &luminal_cuda_lite::host::HostOpContext<'_>,
    ) -> anyhow::Result<Box<dyn luminal_cuda_lite::host::PreparedHostOp>> {
        anyhow::ensure!(
            self.dps,
            "TVM FFI GEMM must be converted to DPS before execution"
        );
        // SAFETY: forwarded HostOp contract; native preparation validates sizes/layouts.
        unsafe { crate::native::prepare(&self.library, ctx) }
    }
}

#[derive(Debug)]
struct GemmMatcher {
    library: Rc<Library>,
}

impl OpMatcher for GemmMatcher {
    fn egglog_constructor(&self) -> &'static str {
        GEMM_CONSTRUCTOR
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 2)]
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LayoutOpConstructors,
                text: "(constructor LayoutTensorOpTvmFfiGemm (LayoutTensor LayoutTensor Layout) LayoutTensorOp)",
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("match.egg"),
            },
        ]
    }

    fn extract(&self, _: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(TvmFfiGemm {
            library: self.library.clone(),
            dps: false,
        })
    }
}

/// Register one GEMM exported by `TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm, ...)`.
///
/// The library must implement the contract documented in this crate's README.
/// Loading is lazy, during HostOp preparation. Add this row to the chosen CUDA
/// registry; registration does not replace or disable any existing implementation.
pub fn gemm_registration(library: impl Into<PathBuf>) -> RegisteredOp {
    let library = Rc::new(Library {
        path: library.into(),
        #[cfg(feature = "device")]
        loaded: std::cell::RefCell::new(None),
    });
    RegisteredOp::new(
        Box::new(GemmMatcher {
            library: library.clone(),
        }),
        Box::new(TvmFfiGemm {
            library,
            dps: false,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::{GEMM_CONSTRUCTOR, GEMM_LABEL, GEMM_WORKSPACE_BYTES, gemm_registration};
    use luminal::layout_ir::{AliasInfo, Sharing};
    use luminal::prelude::*;
    use luminal_cuda_lite::{CudaRuntime, as_host_op, cuda_registry_without_cublaslt};

    #[test]
    fn dps_declares_reads_writes_and_destination_alias() {
        let row = gemm_registration("not-loaded-during-analysis.so");
        assert_eq!(row.constructor(), GEMM_CONSTRUCTOR);
        assert_eq!(row.label(), GEMM_LABEL);
        assert_eq!(row.prototype.label(), GEMM_LABEL);
        assert!(row.prototype.alias_info().is_empty());
        let dps = row
            .prototype
            .to_dps()
            .expect("functional GEMM needs a destination");
        assert_eq!(
            dps.alias_info(),
            vec![AliasInfo {
                operand: 2,
                result: 0,
                sharing: Sharing::Must
            }]
        );
        assert!(dps.operand_reads_memory(0));
        assert!(dps.operand_reads_memory(1));
        assert!(!dps.operand_reads_memory(2)); // beta=0: destination contents are undefined.
        assert!(dps.result_writes_memory(0));
        assert!(dps.to_dps().is_none());
        let cloned = dps.clone_bt_box();
        let host =
            as_host_op(cloned.as_ref()).expect("DPS and cloning preserve the HostOp interface");
        assert_eq!(
            host.workspace_bytes(&Default::default()).unwrap(),
            GEMM_WORKSPACE_BYTES
        );
    }

    fn offers_ffi(a_shape: Vec<usize>, b_shape: Vec<usize>, dtype: DType) -> bool {
        let mut graph = Graph::new();
        let a = graph.tensor(a_shape, dtype);
        let b = graph.tensor(b_shape, dtype);
        let _out = a.matmul(b);
        let mut registry = cuda_registry_without_cublaslt();
        registry.push(gemm_registration("not-loaded-during-analysis.so"));
        let runtime =
            CudaRuntime::load_with_registry(&graph, registry).expect("load external registry");
        assert!(runtime.active_allow_list().contains(&GEMM_CONSTRUCTOR));
        runtime
            .saturated_egraph(&Default::default())
            .expect("saturate graph")
            .nodes
            .values()
            .any(|node| node.op == GEMM_CONSTRUCTOR)
    }

    #[test]
    fn matcher_offers_fp32_rank_two_gemm() {
        assert!(offers_ffi(vec![2, 3], vec![3, 4], DType::F32));
    }

    #[test]
    fn matcher_does_not_claim_fp16() {
        assert!(!offers_ffi(vec![2, 3], vec![3, 4], DType::F16));
    }

    #[test]
    fn matcher_does_not_claim_batched_gemm() {
        assert!(!offers_ffi(vec![2, 3, 4], vec![2, 4, 5], DType::F32));
    }
}
