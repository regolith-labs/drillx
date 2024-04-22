mod operator;
mod ops;
mod utils;

use std::time::Instant;

#[cfg(not(feature = "solana"))]
use sha3::{Digest, Keccak256};

use crate::operator::Operator;
pub use crate::utils::difficulty;
use crate::utils::*;

// TODO Debug build flag (print times)

pub fn drillhash(challenge: [u8; 32], nonce: u64, noise: &[u8]) -> [u8; 32] {
    // The drill part (random sequential calculations and memory reads)
    let timer = Instant::now();
    let digest = Operator::new(&challenge, &nonce.to_le_bytes(), noise).drill();
    // println!("digest: {:?}", digest);
    println!("drill in {} nanos", timer.elapsed().as_nanos());

    // The hash part (keccak proof)
    let timer = Instant::now();

    #[cfg(feature = "solana")]
    let x = solana_program::keccak::hashv(&[
        &nonce.to_le_bytes(),
        challenge.as_slice(),
        digest.as_slice(),
    ])
    .0;

    #[cfg(not(feature = "solana"))]
    let x = Keccak256::new()
        .chain_update(nonce.to_le_bytes())
        .chain_update(challenge.as_slice())
        .chain_update(digest.as_slice())
        .finalize()
        .into();

    println!("hash in {} nanos", timer.elapsed().as_nanos());
    x
}
