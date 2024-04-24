mod operator;
mod utils;

use std::time::Instant;

#[cfg(not(feature = "solana"))]
use sha3::{Digest, Keccak256};

use crate::operator::Operator;
pub use crate::utils::*;

// TODO Debug feature flag for print statements

pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8], noise: &[u8]) -> [u8; 32] {
    // The drill part (non-parallelizable digest)
    // let timer = Instant::now();
    let digest = Operator::new(challenge, nonce, noise).drill();
    // println!("drill in {} nanos", timer.elapsed().as_nanos());

    // The hash part (keccak proof)
    // let timer = Instant::now();

    #[cfg(feature = "solana")]
    let x = solana_program::keccak::hashv(&[digest.as_slice()]).0;
    // let x = solana_program::blake3::hashv(&[&[0]]).0;

    #[cfg(not(feature = "solana"))]
    let x = Keccak256::new()
        .chain_update(digest.as_slice())
        .finalize()
        .into();
    // let x = blake3::hash(&[0]).into();

    // println!("hash in {} nanos\n", timer.elapsed().as_nanos());
    // [0; 32]
    x
}
