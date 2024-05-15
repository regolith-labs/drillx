use std::time::Instant;

use drillx::{equix::SolverMemory, Solution};

const TARGET_DIFFICULTY: u32 = 8; // 12; // 8; //10;

fn main() {
    // Current challenge (255s for demo)
    let challenge = [255; 32];

    // Do work
    let work_timer = Instant::now();
    let (hash, nonce) = do_work(challenge);
    println!("work done in {} nanos", work_timer.elapsed().as_nanos());

    // Now proof
    let proof_timer = Instant::now();
    prove_work(challenge, Solution::new(hash.d, nonce.to_le_bytes()));
    println!("proof done in {} nanos", proof_timer.elapsed().as_nanos());
    println!(
        "work took {}x vs proof",
        work_timer.elapsed().as_nanos() / proof_timer.elapsed().as_nanos()
    );
}

// Parallelize
fn do_work(challenge: [u8; 32]) -> (drillx::Hash, u64) {
    let mut memory = SolverMemory::new();
    let mut nonce: u64 = 0;
    loop {
        // Calculate hash
        if let Ok(hx) = drillx::hash_with_memory(&mut memory, &challenge, &nonce.to_le_bytes()) {
            if hx.difficulty() >= TARGET_DIFFICULTY {
                return (hx, nonce);
            }
        }

        // Increment nonce
        nonce += 1;
    }
}

fn prove_work(challenge: [u8; 32], solution: Solution) {
    println!("Hash {:?}", solution.to_hash().h);
    assert!(solution.is_valid(&challenge));
    assert!(solution.to_hash().difficulty() >= TARGET_DIFFICULTY);
}
