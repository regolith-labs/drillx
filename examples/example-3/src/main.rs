use std::time::Instant;

use drillx::{
    difficulty,
    gpu::{drill_hash, gpu_init, set_noise},
    noise::NOISE,
};

fn main() {
    // Initialize gpu
    unsafe {
        gpu_init();
        set_noise(NOISE.as_usize_slice().as_ptr());
    }

    // Current challenge (255s for demo)
    let timer = Instant::now();
    let challenge = [255; 32];
    let nonce = [2; 8];
    let mut gpu_out = [0; 8];
    unsafe {
        single_drill_hash(
            challenge.as_ptr(),
            u64::from_le_bytes(nonce),
            gpu_out.as_mut_ptr(),
        );
    }
    println!("{gpu_nonce:?}");

    // Calculate hash
    let gpu_hash = drillx::hash(&challenge, &gpu_out);
    println!(
        "gpu found hash with difficulty {} in {} seconds: {}",
        difficulty(gpu_hash),
        timer.elapsed().as_secs(),
        bs58::encode(gpu_hash).into_string(),
    );
}
