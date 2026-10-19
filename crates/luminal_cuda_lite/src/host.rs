//! Opaque library operations compiled into captured CUDA child graphs.
use luminal::buffer_tensor_ir::BufferTensorIrOp;

#[cfg(feature = "tvm-ffi")]
mod library;
#[cfg(feature = "tvm-ffi")]
pub use library::TvmFfiLibrary;

#[derive(Debug, Clone, Copy)]
pub struct DeviceRange {
    pub ptr: u64,
    pub bytes: usize,
}

#[cfg(feature = "device")]
pub struct HostOpContext<'a> {
    pub stream: &'a std::sync::Arc<cudarc::driver::CudaStream>,
    pub inputs: &'a [DeviceRange],
    pub dest: DeviceRange,
    pub workspace: DeviceRange,
    pub dims: &'a luminal::shape::DynMap,
    pub operand_info: &'a [luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>],
    pub result_info: &'a [luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>],
}

/// Created only inside an active graph capture. Recording never executes model work.
#[cfg(feature = "device")]
pub struct CaptureCtx<'a> {
    pub(crate) stream: &'a std::sync::Arc<cudarc::driver::CudaStream>,
}
#[cfg(feature = "device")]
impl CaptureCtx<'_> {
    pub fn stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        self.stream
    }

    /// Record an arbitrary TVM FFI function on this capture's CUDA stream.
    /// Arguments and return values follow the library's own packed signature.
    /// The previous TVM FFI stream context is restored on success and error.
    ///
    /// # Safety
    /// The function must support CUDA graph capture. Arguments must obey its
    /// tensor/layout/alias contract; metadata must live through the call and
    /// device allocations through all replays. Keep the function (which owns
    /// its loaded module) in the preparation until the final replay completes.
    #[cfg(feature = "tvm-ffi")]
    pub unsafe fn call_tvm_ffi(
        &self,
        function: &tvm_ffi::Function,
        args: &[tvm_ffi::AnyView<'_>],
    ) -> anyhow::Result<tvm_ffi::Any> {
        let device = tvm_ffi::DLDevice {
            device_type: tvm_ffi::DLDeviceType::kDLCUDA,
            device_id: i32::try_from(self.stream.context().ordinal())?,
        };
        // SAFETY: CaptureCtx borrows the runtime's live stream on this device.
        unsafe {
            tvm_ffi::with_stream(&device, self.stream.cu_stream().cast(), || {
                function.call_packed(args)
            })
        }
        .map_err(|error| anyhow::anyhow!("TVM FFI call during capture: {error}"))
    }
}

#[cfg(feature = "device")]
pub trait PreparedHostOp {
    /// Record GPU work into the active capture. Preparation must already have
    /// loaded modules and resolved host-side descriptors/algorithms.
    /// # Safety
    /// Bound allocations and addresses must remain valid until the last graph
    /// replay finishes. Their contents are only live during this operation:
    /// other graph nodes may use the same ranges at different times.
    unsafe fn record(&self, capture: &CaptureCtx<'_>) -> anyhow::Result<()>;
}

pub trait HostOp: BufferTensorIrOp {
    /// Scratch reserved in the runtime's shared arena, exclusively for this
    /// operation's captured GPU work. Its contents must not be read or written
    /// during preparation or retained after that work completes. Other nodes
    /// and programs may reuse the same bytes. All device scratch comes from here.
    fn workspace_bytes(&self, _bounds: &crate::symbolic::Bounds) -> anyhow::Result<usize> {
        Ok(0)
    }

    /// Dimensions affecting captured library work. None conservatively means
    /// all dimensions; Some(empty) describes a shape-independent capture.
    /// Include every dimension affecting descriptors, pointers, or recorded work.
    fn capture_dims(&self) -> Option<Vec<luminal::shape::Symbol>> {
        None
    }

    /// Prepare a capture-compatible library call without executing it.
    /// # Safety
    /// ctx's ranges must be valid, sized for their descriptors, and obey this
    /// operation's alias contract. Returned state may retain their addresses.
    #[cfg(feature = "device")]
    unsafe fn prepare(&self, ctx: &HostOpContext<'_>) -> anyhow::Result<Box<dyn PreparedHostOp>>;
}

#[cfg(all(test, feature = "tvm-ffi"))]
mod ffi_tests {
    use super::CaptureCtx;
    use cudarc::driver::{CudaContext, sys as cuda};
    use tvm_ffi::tvm_ffi_sys::dlpack::DLTensor;
    use tvm_ffi::{AnyView, DLDataType, DLDevice, DLDeviceType, Function, TensorView};

    #[test]
    fn arbitrary_tensor_and_scalar_arguments_use_capture_stream_and_restore_it() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let device = DLDevice {
            device_type: DLDeviceType::kDLCUDA,
            device_id: context.ordinal() as i32,
        };
        let previous = tvm_ffi::current_stream(&device);
        let expected_stream = stream.cu_stream() as usize;
        // A second signature, independent of cuBLASLt: borrowed tensor metadata
        // plus a scalar, with a non-status return value. No model work runs here.
        let function = Function::from_typed(
            move |tensor: TensorView, bias: i64| -> tvm_ffi::Result<i64> {
                assert_eq!(tvm_ffi::current_stream(&device) as usize, expected_stream);
                Ok(tensor.numel() as i64 + bias)
            },
        );
        let data = [0.0f32; 6];
        let mut shape = [2i64, 3];
        let mut strides = [3i64, 1];
        let descriptor = DLTensor {
            data: data.as_ptr().cast_mut().cast(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            byte_offset: 0,
        };
        // SAFETY: descriptor and all its storage outlive the calls below.
        let tensor = unsafe { TensorView::from_raw(&descriptor) };
        let bias = 7i64;
        stream
            .begin_capture(cuda::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        let capture = CaptureCtx { stream: &stream };
        // SAFETY: this callback only reads live metadata and supports capture.
        let result = unsafe {
            capture.call_tvm_ffi(&function, &[AnyView::from(&tensor), AnyView::from(&bias)])
        };
        let after_success = tvm_ffi::current_stream(&device);
        // A signature error must also restore the caller's stream context.
        let invalid = 3i64;
        let failure = unsafe {
            capture.call_tvm_ffi(&function, &[AnyView::from(&invalid), AnyView::from(&bias)])
        };
        let after_error = tvm_ffi::current_stream(&device);
        let _graph = stream
            .end_capture(
                cuda::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .unwrap();
        assert_eq!(i64::try_from(result.unwrap()).unwrap(), 13);
        assert!(failure.is_err());
        assert_eq!(after_success, previous);
        assert_eq!(after_error, previous);
    }
}
