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
                text: include_str!("constructor.egg"),
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
