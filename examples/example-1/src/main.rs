use std::{fs::File, io::Read, time::Instant};

const TARGET_DIFFICULTY: u32 = 4; // 12; // 8; //10;

fn main() {
    // Current challenge (255s for demo)
    let challenge = [255; 32];

    // Read noise file.
    let mut file = File::open("../../noise.txt").unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    let noise = buffer.as_slice();

    // Do work
    let work_timer = Instant::now();
    let nonce = do_work(challenge, noise);
    println!("work done in {} nanos", work_timer.elapsed().as_nanos());

    // Now proof
    let proof_timer = Instant::now();
    prove_work(challenge, nonce, noise);
    println!("proof done in {} nanos", proof_timer.elapsed().as_nanos());
    println!(
        "work took {}x vs proof",
        work_timer.elapsed().as_nanos() / proof_timer.elapsed().as_nanos()
    );
}

// TODO Parallelize
fn do_work(challenge: [u8; 32], noise: &[u8]) -> u64 {
    let mut nonce: u64 = 0;
    loop {
        // Calculate hash
        let hx = drillx::hash(&challenge, &nonce.to_le_bytes(), noise);

        // Return if difficulty was met
        if drillx::difficulty(hx) >= TARGET_DIFFICULTY {
            break;
        }

        // Increment nonce
        nonce += 1;
    }
    nonce as u64
}

fn prove_work(challenge: [u8; 32], nonce: u64, noise: &[u8]) {
    let candidate = drillx::hash(&challenge, &nonce.to_le_bytes(), noise);
    println!("candidate hash {candidate:?}");
    assert!(drillx::difficulty(candidate) >= TARGET_DIFFICULTY);
}
