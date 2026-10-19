//! Native ABI/stream bridge. All library setup and tensor metadata are prepared
//! outside capture; record only invokes the already-resolved packed function.
use std::ffi::c_void;

use anyhow::{Context, Result, anyhow, ensure};
use cudarc::cublas::CudaBlas;
use luminal::dtype::PlanDtype;
use luminal::layouts::{DecodedLayout, RightMajorContiguousElementLayout};
use luminal_cuda_lite::host::{CaptureCtx, DeviceRange, HostOpContext, PreparedHostOp};
use tvm_ffi::tvm_ffi_sys::{TVMFFIEnvSetStream, dlpack::DLTensor};
use tvm_ffi::{Any, AnyView, DLDataType, DLDevice, DLDeviceType, Function, Module, TypeIndex};

#[derive(Clone)]
pub(crate) struct LoadedFunction {
    _module: Module,
    function: Function,
}

fn load(library: &crate::gemm::Library) -> Result<LoadedFunction> {
    let mut cache = library.loaded.borrow_mut();
    if let Some(loaded) = cache.as_ref() {
        return Ok(loaded.clone());
    }
    let path = library
        .path
        .to_str()
        .context("FFI library path must be UTF-8")?;
    let module = Module::load_from_file(path)
        .map_err(|error| anyhow!("{error}"))
        .with_context(|| format!("loading TVM FFI GEMM library {}", library.path.display()))?;
    let function = module
        .get_function("gemm")
        .map_err(|error| anyhow!("{error}"))
        .context("resolving the exported TVM FFI function 'gemm'")?;
    let loaded = LoadedFunction {
        _module: module,
        function,
    };
    *cache = Some(loaded.clone());
    Ok(loaded)
}

struct TensorDescriptor {
    data: u64,
    device: DLDevice,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
    dtype: DLDataType,
}

impl TensorDescriptor {
    fn matrix(range: DeviceRange, layout: &DecodedLayout, device: DLDevice) -> Result<Self> {
        ensure!(
            layout.dtype == Some(PlanDtype::F32) && layout.width_bits() == 32,
            "TVM FFI GEMM requires FP32 storage"
        );
        ensure!(
            layout.has::<RightMajorContiguousElementLayout>(),
            "TVM FFI GEMM requires a right-major contiguous layout"
        );
        let shape = layout
            .literal_extents()
            .context("GEMM dimensions must be resolved before capture")?;
        ensure!(
            shape.len() == 2 && shape.iter().all(|&d| d > 0 && d <= i32::MAX as usize),
            "TVM FFI GEMM requires positive rank-2 dimensions fitting int32"
        );
        let bytes = shape
            .iter()
            .try_fold(4usize, |size, dim| size.checked_mul(*dim))
            .context("GEMM matrix size overflow")?;
        ensure!(
            range.ptr != 0 && range.ptr.is_multiple_of(4) && range.bytes >= bytes,
            "GEMM matrix range is null, unaligned, or too small"
        );
        Ok(Self {
            data: range.ptr,
            device,
            shape: vec![shape[0] as i64, shape[1] as i64].into_boxed_slice(),
            strides: vec![shape[1] as i64, 1].into_boxed_slice(),
            dtype: DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
        })
    }

    fn raw(&self) -> DLTensor {
        DLTensor {
            data: self.data as usize as *mut c_void,
            device: self.device,
            ndim: self.shape.len() as i32,
            dtype: self.dtype,
            shape: self.shape.as_ptr().cast_mut(),
            strides: self.strides.as_ptr().cast_mut(),
            byte_offset: 0,
        }
    }
}

struct StreamScope {
    device: DLDevice,
    previous: *mut c_void,
}

impl StreamScope {
    fn enter(device: DLDevice, stream: &cudarc::driver::CudaStream) -> Result<Self> {
        let mut previous = std::ptr::null_mut();
        // SAFETY: the runtime supplies a live stream on this CUDA device.
        let status = unsafe {
            TVMFFIEnvSetStream(
                device.device_type as i32,
                device.device_id,
                stream.cu_stream().cast(),
                &mut previous,
            )
        };
        ensure!(
            status == 0,
            "setting TVM FFI stream context failed ({status})"
        );
        Ok(Self { device, previous })
    }
}

impl Drop for StreamScope {
    fn drop(&mut self) {
        // Restore TLS on errors/unwinds too (the pinned SDK's with_stream does not).
        unsafe {
            let _ = TVMFFIEnvSetStream(
                self.device.device_type as i32,
                self.device.device_id,
                self.previous,
                std::ptr::null_mut(),
            );
        }
    }
}

struct PreparedGemm {
    // Keep both the module's code and its function alive through the final replay.
    loaded: LoadedFunction,
    blas: std::cell::RefCell<CudaBlas>,
    tensors: [TensorDescriptor; 4],
    device: DLDevice,
}

impl PreparedHostOp for PreparedGemm {
    unsafe fn record(&self, capture: &CaptureCtx<'_>) -> Result<()> {
        let _stream_scope = StreamScope::enter(self.device, capture.stream())?;
        let blas = self
            .blas
            .try_borrow_mut()
            .context("GEMM handle is already in use")?;
        let handle = *blas.handle() as usize as i64;
        let tensors = self.tensors.each_ref().map(TensorDescriptor::raw);
        let packed: Vec<Any> = tensors
            .iter()
            .map(|tensor| {
                let mut raw = tvm_ffi::TVMFFIAny::new();
                raw.type_index = TypeIndex::kTVMFFIDLTensorPtr as i32;
                raw.data_union.v_ptr = std::ptr::from_ref(tensor).cast_mut().cast();
                // SAFETY: borrowed descriptors remain live during call_packed. Their
                // device storage is owned by the runtime until every graph replay ends.
                unsafe { Any::from_raw_ffi_any(raw) }
            })
            .collect();
        let mut args = vec![AnyView::from(&handle)];
        args.extend(packed.iter().map(AnyView::from));
        self.loaded
            .function
            .call_packed(&args)
            .map_err(|error| anyhow!("{error}"))
            .context("recording the TVM FFI GEMM call")?;
        Ok(())
    }
}

pub(crate) unsafe fn prepare(
    library: &crate::gemm::Library,
    ctx: &HostOpContext<'_>,
) -> Result<Box<dyn PreparedHostOp>> {
    ensure!(
        ctx.inputs.len() == 2 && ctx.operand_info.len() == 3 && ctx.result_info.len() == 1,
        "TVM FFI GEMM expects A, B, one DPS destination, and one result"
    );
    ensure!(
        ctx.workspace.ptr != 0
            && ctx.workspace.ptr.is_multiple_of(256)
            && ctx.workspace.bytes >= crate::GEMM_WORKSPACE_BYTES,
        "TVM FFI GEMM workspace is missing, too small, or unaligned"
    );
    let device = DLDevice {
        device_type: DLDeviceType::kDLCUDA,
        device_id: i32::try_from(ctx.stream.context().ordinal())?,
    };
    let a = TensorDescriptor::matrix(ctx.inputs[0], &ctx.operand_info[0].layout, device)?;
    let b = TensorDescriptor::matrix(ctx.inputs[1], &ctx.operand_info[1].layout, device)?;
    let out = TensorDescriptor::matrix(ctx.dest, &ctx.result_info[0].layout, device)?;
    ensure!(
        a.shape[1] == b.shape[0] && out.shape[0] == a.shape[0] && out.shape[1] == b.shape[1],
        "incompatible TVM FFI GEMM dimensions"
    );
    let workspace = TensorDescriptor {
        data: ctx.workspace.ptr,
        device,
        shape: vec![i64::try_from(ctx.workspace.bytes)?].into_boxed_slice(),
        strides: vec![1].into_boxed_slice(),
        dtype: DLDataType {
            code: 1,
            bits: 8,
            lanes: 1,
        },
    };
    let loaded = load(library)?;
    let blas = CudaBlas::new(ctx.stream.clone()).context("creating the GEMM cuBLAS handle")?;
    // Module loading, handle initialization, and descriptors are ready. No GEMM
    // or workspace access occurs here: only record() submits model work.
    Ok(Box::new(PreparedGemm {
        loaded,
        blas: std::cell::RefCell::new(blas),
        tensors: [a, b, out, workspace],
        device,
    }))
}
