mod operator;
mod operator2;
mod utils;

use std::slice;
use std::time::Instant;

use periodic_array::PeriodicArray;
use rand::{thread_rng, Fill};
#[cfg(not(feature = "solana"))]
use sha3::{Digest, Keccak256};

use crate::operator::Operator;
use crate::operator2::Operator2;
pub use crate::utils::*;

// TODO Debug feature flag for print statements

pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8], noise: &[u8]) -> [u8; 32] {
    // The drill part (non-parallelizable digest)
    let timer = Instant::now();
    let usize_slice: &[usize] =
        unsafe { slice::from_raw_parts(noise.as_ptr() as *const usize, 32) };
    let n = PeriodicArray::<usize, 32>::new(usize_slice[0..32].try_into().unwrap());
    let digest = Operator2::drill(&n);

    // let digest = Operator2::new(challenge, nonce, noise).drill();
    println!("drill in {} nanos", timer.elapsed().as_nanos());

    // The hash part (keccak proof)
    let timer = Instant::now();
    #[cfg(feature = "solana")]
    let x = solana_program::keccak::hashv(&[digest.as_slice()]).0;

    #[cfg(not(feature = "solana"))]
    let x = Keccak256::new()
        .chain_update(digest.as_slice())
        .finalize()
        .into();

    println!("hash in {} nanos\n", timer.elapsed().as_nanos());
    x
}

pub fn invalid_noise<const N: usize>(noise: &[usize; N]) -> bool {
    // any returns false if noise.len() == 0, so empty noise is invalid
    noise
        .iter()
        .enumerate()
        .any(|(i, n)| i == (*n % noise.len()))
}
