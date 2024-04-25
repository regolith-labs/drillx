use std::time::Instant;

use periodic_array::PeriodicArray;
use sha3::{Digest, Keccak256};

/// Global state for drilling algorithm
#[derive(Debug)]
pub struct Operator2 {}

#[repr(C, align(8))]
pub struct Challenge {
    inner: [u8; 32],
}

impl Challenge {
    pub fn indices(&self) -> [usize; 8] {
        core::array::from_fn(|i| {
            usize::from_le_bytes([
                self.inner[4 * i],
                self.inner[4 * i + 1],
                self.inner[4 * i + 2],
                self.inner[4 * i + 3],
                0,
                0,
                0,
                0,
            ])
        })
    }
}
const D: usize = 16;
const R: usize = 1024;

impl Operator2 {
    /// Build digest using unpredictable and non-parallelizable operations
    pub fn drill<const N: usize>(n: &PeriodicArray<usize, N>) -> [u8; D] {
        let mut digest = [0; D];
        let mut challenge = Challenge { inner: [0; 32] };
        let mut hash_time = 0;
        let mut index_time = 0;
        for i in 0..D / 8 {
            let hash_timer = Instant::now();
            #[cfg(not(feature = "solana"))]
            {
                Keccak256::new()
                    .chain_update(&challenge.inner)
                    .finalize_into((&mut challenge.inner).into());
            }

            #[cfg(feature = "solana")]
            {
                challenge.inner = solana_program::keccak::hashv(&[&challenge.inner]).0;
            }
            hash_time += hash_timer.elapsed().as_nanos();

            // Fetch noise
            let is = challenge.indices();
            let index_timer = Instant::now();
            #[rustfmt::skip]
            {
                digest[8 * i + 0] = recurse::<R, N>(n, is[0]);
                digest[8 * i + 1] = recurse::<R, N>(n, is[1]);
                digest[8 * i + 2] = recurse::<R, N>(n, is[2]);
                digest[8 * i + 3] = recurse::<R, N>(n, is[3]);
                digest[8 * i + 4] = recurse::<R, N>(n, is[4]);
                digest[8 * i + 5] = recurse::<R, N>(n, is[5]);
                digest[8 * i + 6] = recurse::<R, N>(n, is[6]);
                digest[8 * i + 7] = recurse::<R, N>(n, is[7]);
            };
            index_time += index_timer.elapsed().as_nanos();
        }

        println!(
            "eight recurse: hash {hash_time} vs index {index_time}: {}x",
            index_time as f64 / hash_time as f64
        );

        digest
    }
}

#[inline(always)]
fn recurse<const R: usize, const N: usize>(n: &PeriodicArray<usize, N>, mut index: usize) -> u8 {
    let oi = index;
    for _ in 0..R {
        index ^= n[index]; // TODO more complicated stateful thing here
    }
    (n[index] >> (oi % 8)) as u8
}
