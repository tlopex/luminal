//! Follow the graph -> search -> inputs -> execute -> output workflow.
//! Run: cargo run -p luminal_reference --example matmul_walkthrough

use anyhow::{Result, ensure};
use luminal::prelude::*;
use luminal_reference::ReferenceRuntime;

fn main() -> Result<()> {
    // 1. Record the computation. No matrix multiplication happens here.
    let mut graph = Graph::new();
    let a = graph.tensor(vec![2, 3], DType::F32);
    let b = graph.tensor(vec![3, 2], DType::F32);
    let z = a.matmul(b) + 1.0;

    // A = [[1, 2, 3], [4, 5, 6]], B = [[7, 8], [9, 10], [11, 12]].
    let inputs = vec![
        (a.id, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].into()),
        (b.id, vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0].into()),
    ];
    let search_data = inputs.iter().cloned().collect();

    // 2. Load the graph and search for an executable implementation.
    // This example uses the CPU reference runtime, not a CUDA HostOp.
    let mut runtime = ReferenceRuntime::load(&graph)?;
    runtime.search(
        &Default::default(),
        &graph.dyn_map,
        &search_data,
        &luminal_reference::harness_search_options(),
    )?;

    // 3. Supply the actual inputs for execution.
    for (id, value) in inputs {
        runtime.set_data(id, value);
    }

    // 4. Execute the selected implementation and read the result.
    runtime.execute()?;
    let actual = runtime.get_f32(z.id)?;
    let expected = [59.0f32, 65.0, 140.0, 155.0];
    ensure!(actual.len() == expected.len(), "unexpected output shape");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            (actual - expected).abs() < 1e-5,
            "output[{index}]: expected {expected}, got {actual}"
        );
    }

    println!("A: [[1, 2, 3], [4, 5, 6]]");
    println!("B: [[7, 8], [9, 10], [11, 12]]");
    println!("Z = A x B + 1: {actual:?}");
    println!("PASS: expected [[59, 65], [140, 155]]");
    Ok(())
}
