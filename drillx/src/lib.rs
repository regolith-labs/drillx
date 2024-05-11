mod operator2;
mod utils;

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod noise;

#[cfg(feature = "benchmark")]
use std::time::Instant;

// use crate::operator::Operator;
use crate::operator2::Operator2;
pub use crate::utils::*;

// TODO Debug feature flag for print statements

#[cfg(not(feature = "benchmark"))]
pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8]) -> [u8; 32] {
    let digest = Operator2::new(challenge, nonce).drill();
    solana_program::keccak::hashv(&[digest.as_slice()]).0
}

#[cfg(feature = "benchmark")]
pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8]) -> [u8; 32] {
    // The drill part (non-parallelizable digest)
    println!("Nonce {}", u64::from_le_bytes(*nonce));
    let timer = Instant::now();
    let digest = Operator2::new(challenge, nonce).drill();
    println!("drill in {} nanos", timer.elapsed().as_nanos());

    // The hash part (keccak proof)
    let timer = Instant::now();
    let x = solana_program::keccak::hashv(&[digest.as_slice()]).0;
    println!("hash in {} nanos\n", timer.elapsed().as_nanos());
    x
}
