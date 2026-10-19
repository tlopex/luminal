//! Rust -> TVM FFI -> a C++ cuBLAS wrapper -> GPU.
use anyhow::{Context, Result, anyhow, ensure};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, sys as cuda};
use std::ffi::c_void;
use std::sync::Arc;
use tvm_ffi::tvm_ffi_sys::{TVMFFIEnvGetStream, TVMFFIEnvSetStream, dlpack::DLTensor};
use tvm_ffi::{Any, AnyView, DLDataType, DLDevice, DLDeviceType, Function, Module, TypeIndex};

const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

struct Matrix {
    storage: CudaSlice<f32>,
    shape: [i64; 2],
    strides: [i64; 2],
}

impl Matrix {
    fn new(stream: &Arc<CudaStream>, rows: usize, cols: usize, values: &[f32]) -> Result<Self> {
        ensure!(rows > 0 && cols > 0 && values.len() == rows * cols);
        Ok(Self {
            storage: stream.clone_htod(values)?,
            shape: [i64::try_from(rows)?, i64::try_from(cols)?],
            strides: [i64::try_from(cols)?, 1],
        })
    }

    // Prepare descriptors before capture: cudarc's pointer access can record events.
    // The matrix, metadata, and allocation must outlive every use of this descriptor.
    fn descriptor(&self, stream: &CudaStream, device: DLDevice) -> DLTensor {
        DLTensor {
            data: self.storage.device_ptr(stream).0 as usize as *mut c_void,
            device,
            ndim: 2,
            dtype: DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            shape: self.shape.as_ptr().cast_mut(),
            strides: self.strides.as_ptr().cast_mut(),
            byte_offset: 0,
        }
    }
}

// The 0.1.12 Rust with_stream helper does not restore TLS on an error.
// This scope restores it on success, error, and unwinding without changing TVM FFI.
struct StreamScope {
    device: DLDevice,
    previous: *mut c_void,
}

impl StreamScope {
    fn enter(device: DLDevice, stream: &CudaStream) -> Result<Self> {
        let mut previous = std::ptr::null_mut();
        // SAFETY: this device and stream come from the live CUDA context in main.
        let status = unsafe {
            TVMFFIEnvSetStream(
                device.device_type as i32,
                device.device_id,
                stream.cu_stream().cast(),
                &mut previous,
            )
        };
        ensure!(status == 0, "TVMFFIEnvSetStream failed: {status}");
        Ok(Self { device, previous })
    }
}

impl Drop for StreamScope {
    fn drop(&mut self) {
        // SAFETY: restore the original thread-local handle; this scope owns no stream.
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

/// Invoke the exported function, with borrowed DLTensor arguments and no data copy.
///
/// # Safety
/// Descriptors must refer to live, adequately sized device allocations and metadata.
/// All GPU access is serialized on stream; the handle must belong to its device.
/// Those resources must outlive asynchronous work, including every graph replay.
unsafe fn call_gemm(
    function: &Function,
    blas: &mut CudaBlas,
    stream: &CudaStream,
    device: DLDevice,
    tensors: &[DLTensor; 4],
) -> Result<()> {
    let _scope = StreamScope::enter(device, stream)?;
    let tensor_args: Vec<Any> = tensors
        .iter()
        .map(|tensor| {
            let mut raw = tvm_ffi::TVMFFIAny::new();
            raw.type_index = TypeIndex::kTVMFFIDLTensorPtr as i32;
            raw.data_union.v_ptr = std::ptr::from_ref(tensor).cast_mut().cast();
            // SAFETY: type 7 is a non-owning DLTensor pointer. The descriptors and
            // metadata stay live for this call, and Any does not free their data.
            unsafe { Any::from_raw_ffi_any(raw) }
        })
        .collect();
    let handle = *blas.handle() as usize as i64;
    let mut args = vec![AnyView::from(&handle)];
    args.extend(tensor_args.iter().map(AnyView::from));

    // This is the actual TVM FFI call boundary. The function comes from gemm.so.
    function
        .call_packed(&args)
        .map_err(|error| anyhow!("{error}"))?;
    Ok(())
}

fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    (0..m * n)
        .map(|index| {
            let (row, col) = (index / n, index % n);
            (0..k)
                .map(|inner| a[row * k + inner] * b[inner * n + col])
                .sum()
        })
        .collect()
}

fn check_close(actual: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(actual.len() == expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            actual.is_finite() && (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0),
            "output[{index}]: expected {expected}, got {actual}"
        );
    }
    Ok(())
}

fn expect_error(name: &str, result: Result<()>, message: &str) -> Result<()> {
    let error = result.expect_err("invalid arguments must be rejected");
    ensure!(
        error.to_string().contains(message),
        "{name}: unexpected error: {error}"
    );
    println!("PASS validation: {name}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    function: &Function,
    blas: &mut CudaBlas,
    stream: &Arc<CudaStream>,
    device: DLDevice,
    workspace: &DLTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    let a_values: Vec<f32> = (1..=m * k).map(|x| x as f32).collect();
    let b_values: Vec<f32> = (m * k + 1..=m * k + k * n).map(|x| x as f32).collect();
    let mut a = Matrix::new(stream, m, k, &a_values)?;
    let mut b = Matrix::new(stream, k, n, &b_values)?;
    let mut out = Matrix::new(stream, m, n, &vec![f32::NAN; m * n])?;
    let tensors = [
        a.descriptor(stream, device),
        b.descriptor(stream, device),
        out.descriptor(stream, device),
        workspace.clone(),
    ];
    stream.synchronize()?;

    // Capture BEFORE the first GEMM: HostOp preparation cannot execute model work.
    stream.begin_capture(cuda::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    // End capture even if the FFI call reports an error.
    // SAFETY: allocations and metadata outlive this graph and all its replays.
    let recorded = unsafe { call_gemm(function, blas, stream, device, &tensors) };
    let ended = stream.end_capture(
        cuda::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    recorded?;
    let graph = ended?.context("FFI call produced no captured graph")?;
    ensure!(
        stream.clone_dtoh(&out.storage)?.iter().all(|x| x.is_nan()),
        "capture executed GEMM"
    );

    // Also exercise the ordinary FFI call outside capture.
    unsafe { call_gemm(function, blas, stream, device, &tensors)? };
    stream.synchronize()?;
    let actual = stream.clone_dtoh(&out.storage)?;
    check_close(&actual, &reference(&a_values, &b_values, m, k, n))?;
    ensure!(stream.clone_dtoh(&a.storage)? == a_values, "A was modified");
    ensure!(stream.clone_dtoh(&b.storage)? == b_values, "B was modified");
    println!("PASS eager {m}x{k} @ {k}x{n}: {actual:?}; inputs unchanged");

    // Replay must read current inputs, not values seen while recording the graph.
    stream.memcpy_htod(&vec![f32::NAN; m * n], &mut out.storage)?;
    for replay in 0..2 {
        let next_a: Vec<f32> = a_values
            .iter()
            .map(|x| x * (replay + 2) as f32 + 0.5)
            .collect();
        let next_b: Vec<f32> = b_values
            .iter()
            .map(|x| x - (replay + 1) as f32 * 0.25)
            .collect();
        stream.memcpy_htod(&next_a, &mut a.storage)?;
        stream.memcpy_htod(&next_b, &mut b.storage)?;
        graph.launch()?;
        stream.synchronize()?;
        check_close(
            &stream.clone_dtoh(&out.storage)?,
            &reference(&next_a, &next_b, m, k, n),
        )?;
        ensure!(
            stream.clone_dtoh(&a.storage)? == next_a,
            "replay modified A"
        );
        ensure!(
            stream.clone_dtoh(&b.storage)? == next_b,
            "replay modified B"
        );
    }
    println!("PASS capture {m}x{k} @ {k}x{n}: no GEMM warmup; two replays with changed inputs");

    let mut bad_dtype = tensors.clone();
    bad_dtype[0].dtype.code = 1;
    expect_error(
        "wrong dtype",
        unsafe { call_gemm(function, blas, stream, device, &bad_dtype) },
        "wrong dtype",
    )?;

    let wrong_b = Matrix::new(stream, k + 1, n, &vec![0.0; (k + 1) * n])?;
    let mut bad_shape = tensors.clone();
    bad_shape[1] = wrong_b.descriptor(stream, device);
    stream.synchronize()?;
    expect_error(
        "incompatible shapes",
        unsafe { call_gemm(function, blas, stream, device, &bad_shape) },
        "incompatible GEMM shapes",
    )?;

    // Only test aliasing when A's allocation is large enough for the output descriptor.
    if k >= n {
        let mut alias = tensors.clone();
        alias[2].data = alias[0].data;
        expect_error(
            "output aliases input",
            unsafe { call_gemm(function, blas, stream, device, &alias) },
            "out must not alias",
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let library = std::env::args()
        .nth(1)
        .context("usage: tvm-ffi-gemm-demo /path/to/gemm.so")?;
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?; // Non-default stream, as in a captured HostOp.
    let mut blas = CudaBlas::new(stream.clone())?; // Initialize outside graph capture.
    let device = DLDevice {
        device_type: DLDeviceType::kDLCUDA,
        device_id: context.ordinal() as i32,
    };
    let workspace_storage = stream.alloc_zeros::<u8>(WORKSPACE_BYTES)?;
    let mut workspace_shape = [WORKSPACE_BYTES as i64];
    let mut workspace_strides = [1i64];
    let workspace = DLTensor {
        data: workspace_storage.device_ptr(&stream).0 as usize as *mut c_void,
        device,
        ndim: 1,
        dtype: DLDataType {
            code: 1,
            bits: 8,
            lanes: 1,
        },
        shape: workspace_shape.as_mut_ptr(),
        strides: workspace_strides.as_mut_ptr(),
        byte_offset: 0,
    };
    stream.synchronize()?;

    // Keep the module loaded until its function and all captured graphs are finished.
    let module = Module::load_from_file(&library).map_err(|error| anyhow!("{error}"))?;
    let gemm = module
        .get_function("gemm")
        .map_err(|error| anyhow!("{error}"))?;
    println!("Loaded {library}: gemm via TVM FFI");
    // SAFETY: querying thread-local stream state does not dereference its handle.
    let original_stream =
        unsafe { TVMFFIEnvGetStream(device.device_type as i32, device.device_id) };
    for (m, k, n) in [(2, 3, 2), (3, 2, 4), (1, 5, 3), (4, 1, 2)] {
        run_case(&gemm, &mut blas, &stream, device, &workspace, m, k, n)?;
        ensure!(
            unsafe { TVMFFIEnvGetStream(device.device_type as i32, device.device_id) }
                == original_stream,
            "FFI stream context was not restored after a call or error"
        );
    }
    stream.synchronize()?;
    println!(
        "PASS: Rust -> TVM FFI -> cuBLAS; eager, capture/replay, validation, stream restoration"
    );
    Ok(())
}
