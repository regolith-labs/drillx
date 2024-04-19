use std::{fs::File, io::Read, time::Instant};

use num_bigint::BigInt;
use num_traits::{FromBytes, ToPrimitive};
use sha3::{Digest, Keccak256};

const TARGET_DIFFICULTY: u32 = 19; // 10; // 8; // 4; //10;

fn main() {
    // Current challenge (255s for demo)
    let challenge = [255; 32];

    // Read noise file.
    let mut file = File::open("noise.txt").unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    let noise = buffer.as_slice();
    println!("noise len {}", noise.len());

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

// TODO Parallelize
fn do_work(challenge: [u8; 32], noise: &[u8]) -> u64 {
    let timer = Instant::now();
    let mut nonce = 0;
    loop {
        // Calculate hash
        let solution = drill_hash(challenge, nonce, noise);

        // Return if difficulty was met
        if difficulty(solution) >= TARGET_DIFFICULTY {
            break;
        }

        // Increment nonce
        nonce += 1;
    }

    println!(
        "{} hashes in {} sec ({} H/s)",
        nonce,
        timer.elapsed().as_secs(),
        nonce / timer.elapsed().as_secs()
    );

    nonce as u64
}

fn drill_hash(challenge: [u8; 32], nonce: u64, noise: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new()
        .chain_update(nonce.to_le_bytes())
        .chain_update(challenge.as_ref());

    // The drill part (1024 sequential modpow and mem reads)
    // let timer = Instant::now();
    let len = BigInt::from(noise.len());
    let mut digest = [0u8; 1024];
    let mut addr = BigInt::from_le_bytes(&challenge);
    let mut n = BigInt::from(nonce.saturating_add(2));
    for i in 0..1024 {
        addr = addr.modpow(&n, &len);
        digest[i] = noise[addr.to_usize().unwrap()];
        n = BigInt::from(digest[i].saturating_add(2));
    }
    // println!("reads in {} nanos", timer.elapsed().as_nanos());

    // The hash part (keccak proof)
    // let timer = Instant::now();
    hasher.update(digest.as_slice());
    let x = hasher.finalize().into();
    // println!("hash in {} nanos", timer.elapsed().as_nanos());
    x
}

fn prove_work(challenge: [u8; 32], nonce: u64, noise: &[u8]) -> bool {
    let candidate = drill_hash(challenge, nonce, noise);
    println!("candidate hash {candidate:?}");
    difficulty(candidate) >= TARGET_DIFFICULTY
}

fn difficulty(hash: [u8; 32]) -> u32 {
    let mut count = 0;
    for &byte in &hash {
        let lz = byte.leading_zeros();
        count += lz;
        if lz < 8 {
            break;
        }
    }
    count
}
