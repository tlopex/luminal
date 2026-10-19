//! Argument adaptation for FlashInfer's upstream TVM FFI GEMM functions.
use anyhow::{Context, Result, anyhow, ensure};
use cudarc::cublaslt::{result as lt, sys};
use luminal::dtype::PlanDtype;
use luminal::layouts::RightMajorContiguousElementLayout;
use tvm_ffi::tvm_ffi_sys::dlpack::{DLDataTypeCode, DLTensor};
use tvm_ffi::{Any, AnyView, DLDataType, DLDevice, DLDeviceType, Function, Tensor, TensorView};

use crate::host::{CaptureCtx, DeviceRange, HostOpContext, PreparedHostOp, TvmFfiLibrary};

struct LtHandle(sys::cublasLtHandle_t);
impl Drop for LtHandle {
    fn drop(&mut self) {
        // SAFETY: uniquely owned; the prepared operation outlives its graph.
        unsafe {
            let _ = lt::destroy_handle(self.0);
        }
    }
}

struct PreparedGemm {
    function: Function,
    handle: LtHandle,
    algorithms: Tensor,
    ranges: [DeviceRange; 4],
    shapes: [Vec<i64>; 4],
    device: DLDevice,
}

pub(super) unsafe fn prepare(
    library: &TvmFfiLibrary,
    ctx: &HostOpContext<'_>,
) -> Result<Box<dyn PreparedHostOp>> {
    ensure!(
        ctx.inputs.len() == 2 && ctx.result_info.len() == 1,
        "FlashInferGemm arity"
    );
    let slots = [
        &ctx.operand_info[0],
        &ctx.operand_info[1],
        &ctx.result_info[0],
    ];
    let ranges = [ctx.inputs[0], ctx.inputs[1], ctx.dest, ctx.workspace];
    let mut shapes = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        vec![i64::try_from(ctx.workspace.bytes)?],
    ];
    for (i, slot) in slots.iter().enumerate() {
        slot.layout
            .require::<RightMajorContiguousElementLayout>("FlashInferGemm")?;
        let dtype = if i < 2 {
            PlanDtype::Bf16
        } else {
            PlanDtype::F32
        };
        ensure!(slot.layout.dtype == Some(dtype), "FlashInferGemm dtype");
        let shape = slot
            .layout
            .literal_extents()
            .context("FlashInferGemm unresolved geometry")?;
        ensure!(
            shape.len() == 2 && shape.iter().all(|&d| d > 0 && d <= i32::MAX as usize),
            "FlashInferGemm requires positive int32 matrix extents"
        );
        let bytes = shape
            .iter()
            .try_fold(if i < 2 { 2usize } else { 4usize }, |size, d| {
                size.checked_mul(*d)
            })
            .context("FlashInferGemm size overflow")?;
        ensure!(ranges[i].bytes >= bytes, "FlashInferGemm buffer too small");
        shapes[i] = shape.into_iter().map(|d| d as i64).collect();
    }
    ensure!(
        shapes[0][1] == shapes[1][1] && shapes[2] == [shapes[0][0], shapes[1][0]],
        "FlashInferGemm shape mismatch"
    );
    ctx.stream.context().bind_to_thread()?;
    let module = library.load().context("loading FlashInfer")?;
    let get_algos = module
        .get_function("mm_bf16_cublaslt_get_algos")
        .map_err(|e| anyhow!("resolving FlashInfer algorithm query: {e}"))?;
    let prepared = PreparedGemm {
        function: module
            .get_function("mm_bf16_cublaslt_run_with_algo")
            .map_err(|e| anyhow!("resolving FlashInfer GEMM: {e}"))?,
        handle: LtHandle(
            lt::create_handle().map_err(|e| anyhow!("FlashInfer cuBLASLt handle: {e:?}"))?,
        ),
        // Cache one heuristic algorithm on the CPU. No GPU allocation or model
        // execution occurs in preparation; scratch is only sized by the query.
        algorithms: Tensor::from_slice(
            &[0u8; std::mem::size_of::<sys::cublasLtMatmulAlgo_t>()],
            &[std::mem::size_of::<sys::cublasLtMatmulAlgo_t>() as i64],
        )
        .map_err(|e| anyhow!("FlashInfer algorithm buffer: {e}"))?,
        ranges,
        shapes,
        device: DLDevice {
            device_type: DLDeviceType::kDLCUDA,
            device_id: i32::try_from(ctx.stream.context().ordinal())?,
        },
    };
    let count: i64 = prepared
        .with_args(|args| {
            get_algos
                .call_packed(&args[..7])
                .map_err(|e| anyhow!("FlashInfer algorithm query: {e}"))
        })?
        .try_into()
        .map_err(|e| anyhow!("FlashInfer algorithm count: {e}"))?;
    ensure!(
        count > 0,
        "FlashInfer found no GEMM algorithm within the configured scratch budget"
    );
    Ok(Box::new(prepared))
}

impl PreparedGemm {
    fn with_args(&self, call: impl FnOnce(&[AnyView<'_>; 8]) -> Result<Any>) -> Result<Any> {
        let dtypes = [
            DLDataType::new(DLDataTypeCode::kDLBfloat, 16, 1),
            DLDataType::new(DLDataTypeCode::kDLBfloat, 16, 1),
            DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1),
            DLDataType::new(DLDataTypeCode::kDLUInt, 8, 1),
        ];
        let descriptors: [DLTensor; 4] = std::array::from_fn(|i| DLTensor {
            data: self.ranges[i].ptr as *mut _,
            device: self.device,
            ndim: self.shapes[i].len() as i32,
            dtype: dtypes[i],
            shape: self.shapes[i].as_ptr().cast_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        });
        // SAFETY: descriptors and their storage outlive the packed call. The
        // provider reads metadata; GPU buffers obey HostOp::prepare's contract.
        let tensors = descriptors
            .each_ref()
            .map(|desc| unsafe { TensorView::from_raw(desc) });
        let handle = self.handle.0 as i64;
        let algo_idx = 0i64;
        call(&[
            AnyView::from(&tensors[0]),
            AnyView::from(&tensors[1]),
            AnyView::default(), // Optional bias: None.
            AnyView::from(&tensors[2]),
            AnyView::from(&tensors[3]),
            AnyView::from(&handle),
            AnyView::from(&self.algorithms),
            AnyView::from(&algo_idx),
        ])
    }
}

impl PreparedHostOp for PreparedGemm {
    unsafe fn record(&self, capture: &CaptureCtx<'_>) -> Result<()> {
        self.with_args(|args| {
            // SAFETY: the cached algorithm needs no GPU allocation; the
            // provider obtains capture's stream through TVM FFI's stream scope.
            unsafe { capture.call_tvm_ffi(&self.function, args) }
        })?;
        Ok(())
    }
}
