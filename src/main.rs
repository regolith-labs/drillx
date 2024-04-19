use std::{fs::File, io::Read, time::Instant};

use num_bigint::BigInt;
use num_traits::{FromBytes, ToPrimitive};
use sha3::{Digest, Keccak256};

const TARGET_DIFFICULTY: u32 = 4; //10;

fn main() {
    // Current challenge (255s for demo)
    let challenge = [255; 32];

    // Read noise file.
    let mut file = File::open("noise.txt").unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    let noise = buffer.as_slice();
    println!("noise len {}", noise.len());
    println!("noise: {:?}", &noise[0..8]);

    // Do work
    let work_timer = Instant::now();
    let nonce = do_work(challenge, noise);
    println!("work done in {} nanos", work_timer.elapsed().as_nanos());

    // Now proof
    let proof_timer = Instant::now();
    assert!(prove_work(challenge, nonce, noise));
    println!("proof done in {} nanos", proof_timer.elapsed().as_nanos());

    println!(
        "work took {}x vs proof",
        work_timer.elapsed().as_nanos() / proof_timer.elapsed().as_nanos()
    );
}

fn do_work(challenge: [u8; 32], noise: &[u8]) -> u64 {
    let mut nonce = 0;
    loop {
        // Require every nonce to have a sequential work component
        let solution = memhash(challenge, nonce, noise);

        // Update hasher (digest 32 bytes and update internal state)
        let d = difficulty(solution);
        if d >= TARGET_DIFFICULTY {
            break;
        }

        // Increment nonce
        nonce += 1;
    }

    nonce as u64
}

fn memhash(challenge: [u8; 32], nonce: usize, noise: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new()
        .chain_update(nonce.to_le_bytes())
        .chain_update(challenge.as_ref());

    let timer = Instant::now();
    let len = BigInt::from(noise.len());
    let mut digest = [0u8; 1024];
    let mut addr = BigInt::from_le_bytes(&challenge);
    let mut n = BigInt::from(nonce.saturating_add(2));
    for i in 0..1024 {
        addr = addr.modpow(&n, &len);
        digest[i] = noise[addr.to_usize().unwrap()];
        n = BigInt::from(digest[i].saturating_add(2));
    }
    println!("reads in {} nanos", timer.elapsed().as_nanos());

    let timer = Instant::now();
    hasher.update(digest.as_slice());
    println!("digest in {} nanos", timer.elapsed().as_nanos());

    let timer = Instant::now();
    let x = hasher.finalize().into();
    println!("finalized in {} nanos", timer.elapsed().as_nanos());
    x
}

fn prove_work(challenge: [u8; 32], nonce: u64, noise: &[u8]) -> bool {
    let candidate = memhash(challenge, nonce as usize, noise);
    println!("candidate hash = {candidate:?}");
    difficulty(candidate) >= TARGET_DIFFICULTY
}

fn difficulty(hash: [u8; 32]) -> u32 {
    let mut count = 0;
    for &byte in &hash {
        // Count leading zeros in current byte
        let lz = byte.leading_zeros();
        count += lz;
        // If lz is less than 8, it means we've encountered a non-zero bit
        if lz < 8 {
            break;
        }
    }
    count
}
