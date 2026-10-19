use luminal::layout_ir::{AliasInfo, Sharing};
use luminal::prelude::*;
use luminal_cuda_lite::{CudaRuntime, as_host_op, cuda_registry_without_cublaslt};
use luminal_tvm_ffi::{GEMM_CONSTRUCTOR, GEMM_LABEL, GEMM_WORKSPACE_BYTES, gemm_registration};

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
    let host = as_host_op(cloned.as_ref()).expect("DPS and cloning preserve the HostOp interface");
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
