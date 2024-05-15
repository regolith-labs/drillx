use std::time::Instant;

const TEST_SIZE: u64 = 1000;

fn main() {
    println!("Benchmarking...");
    let mut memory = drillx::equix::SolverMemory::new();
    let challenge = [255; 32];
    let timer = Instant::now();
    for nonce in 0..TEST_SIZE {
        drillx::hash_with_memory(&mut memory, &challenge, &nonce.to_le_bytes()).ok();
    }
    println!(
        "Did {} hashes in {} ms\nHashrate: {} H/s",
        TEST_SIZE,
        timer.elapsed().as_millis(),
        (TEST_SIZE as u128)
            .saturating_mul(1000)
            .saturating_div(timer.elapsed().as_millis())
    );
}
