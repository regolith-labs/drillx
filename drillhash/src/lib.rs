mod ops;
mod utils;

use std::time::Instant;

use sha3::{Digest, Keccak256};
use strum::IntoEnumIterator;

use crate::ops::*;
pub use crate::utils::difficulty;
use crate::utils::*;

// TODO Solana feature flag
// TODO Debug build flag (print times)

pub fn drillhash(challenge: [u8; 32], nonce: u64, noise: &[u8]) -> [u8; 32] {
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

fn drill(challenge: [u8; 32], nonce: u64, noise: &[u8]) -> [u8; 32] {
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
        while !random_op(ops, &mut addr, challenge, nonce_, noise)
            .op(&mut addr, challenge, nonce_, noise)
        {
            // println!("{:?} {}", op, addr);
            // Noop
        }

        // Append to digest
        digest[i] = noise[addr as usize % noise.len()];
    }

    // Return
    digest
}
