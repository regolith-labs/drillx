mod ops;
mod utils;

use std::{fs::File, io::Read, time::Instant};

use primes::is_prime;
use sha3::{Digest, Keccak256};
use strum::IntoEnumIterator;

use crate::ops::*;
use crate::utils::*;

const TARGET_DIFFICULTY: u32 = 4; // 8; //10;

fn main() {
    // Current challenge (255s for demo)
    let challenge = [255; 32];

    // Read noise file.
    let mut file = File::open("noise.txt").unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    let noise = buffer.as_slice();
    if !is_prime(noise.len() as u64) {
        panic!("Noise file length must be prime");
    }

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
    nonce as u64
}

fn drill_hash(challenge: [u8; 32], nonce: u64, noise: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new()
        .chain_update(nonce.to_le_bytes())
        .chain_update(challenge.as_ref());

    // The drill part (random sequential calculations and memory reads)
    let timer = Instant::now();
    let digest = drill(challenge, nonce, noise);
    println!("drill in {} nanos", timer.elapsed().as_nanos());

    // The hash part (keccak proof)
    let timer = Instant::now();
    hasher.update(digest.as_slice());
    let x = hasher.finalize().into();
    println!("hash in {} nanos", timer.elapsed().as_nanos());
    x
}

pub fn drill(challenge: [u8; 32], nonce: u64, noise: &[u8]) -> [u8; 32] {
    // Stateful ops
    let ops: &'static mut [RandomOp] = Box::leak(RandomOp::iter().collect::<Box<[_]>>());

    // Generate starting address
    let b = blake3::hash(&[challenge.as_ref(), nonce.to_le_bytes().as_ref()].concat());
    let mut addr = modpow(
        u64::from_le_bytes(b.as_bytes()[0..8].try_into().unwrap()),
        u64::from_le_bytes(b.as_bytes()[8..16].try_into().unwrap()),
        u64::MAX / 2, // len as u64,
    );

    // Build digest
    let nonce_ = nonce.to_le_bytes();
    let mut digest = [0; 32];
    for i in 0..32 {
        // Do random ops on address until exit
        let op = random_op(ops, &mut addr, challenge, nonce_, noise);
        while !op.op(&mut addr, challenge, nonce_, noise) {
            // println!("{:?} {}", op, addr);
            // Noop
        }

        // Append to digest
        digest[i] = noise[addr as usize % noise.len()];
    }

    // Return
    digest
}

fn prove_work(challenge: [u8; 32], nonce: u64, noise: &[u8]) -> bool {
    let candidate = drill_hash(challenge, nonce, noise);
    println!("candidate hash {candidate:?}");
    difficulty(candidate) >= TARGET_DIFFICULTY
}
