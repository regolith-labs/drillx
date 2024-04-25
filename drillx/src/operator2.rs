use std::{mem, time::Instant};

#[cfg(not(feature = "solana"))]
use sha3::{Digest, Keccak256};

/// Digest size
const D: usize = 16;

/// Number of recursive reads per loop
const R: usize = 2048;

/// Global state for drilling algorithm
pub struct Operator2<'a> {
    state: [u8; 32],
    noise: &'a [usize],
}

impl<'a> Operator2<'a> {
    pub fn new(challenge: &'a [u8; 32], nonce: &'a [u8; 8], noise: &'a [u8]) -> Operator2<'a> {
        let state;
        #[cfg(not(feature = "solana"))]
        {
            Keccak256::new()
                .chain_update(&challenge.as_slice())
                .chain_update(&nonce.as_slice())
                .finalize_into((&mut state).into());
        }

        #[cfg(feature = "solana")]
        {
            state = solana_program::keccak::hashv(&[&challenge.as_slice(), &nonce.as_slice()]).0;
        }

        Operator2 {
            noise: as_usize_slice(noise).expect("Failed to read noise as &[usize]"),
            state,
        }
    }

    /// Build digest using unpredictable and non-parallelizable operations
    pub fn drill(&mut self) -> [u8; D] {
        let mut digest = [0; D];
        let mut hash_time = 0;
        let mut index_time = 0;
        for i in 0..D / 8 {
            // Hash state
            let hash_timer = Instant::now();
            self.hash();
            hash_time += hash_timer.elapsed().as_nanos();

            // Fetch noise
            let idx = self.indices();
            let index_timer = Instant::now();
            digest[8 * i + 0] = self.recurse::<R>(idx[0]);
            digest[8 * i + 1] = self.recurse::<R>(idx[1]);
            digest[8 * i + 2] = self.recurse::<R>(idx[2]);
            digest[8 * i + 3] = self.recurse::<R>(idx[3]);
            digest[8 * i + 4] = self.recurse::<R>(idx[4]);
            digest[8 * i + 5] = self.recurse::<R>(idx[5]);
            digest[8 * i + 6] = self.recurse::<R>(idx[6]);
            digest[8 * i + 7] = self.recurse::<R>(idx[7]);
            index_time += index_timer.elapsed().as_nanos();
        }

        println!(
            "eight recurse: hash {hash_time} vs index {index_time}: {}x",
            index_time as f64 / hash_time as f64
        );

        digest
    }

    #[inline(always)]
    fn hash(&mut self) {
        #[cfg(not(feature = "solana"))]
        {
            Keccak256::new()
                .chain_update(&self.state)
                .finalize_into((&mut self.state).into());
        }

        #[cfg(feature = "solana")]
        {
            self.state = solana_program::keccak::hashv(&[&self.state]).0;
        }
    }

    #[inline(always)]
    fn recurse<const R: usize>(&self, mut index: usize) -> u8 {
        let oi = index;
        for _ in 0..R {
            index ^= self.noise(index);
            // TODO stateful op here
        }
        (self.noise(index) >> (oi % 8)) as u8
    }

    #[inline(always)]
    fn noise(&self, addr: usize) -> usize {
        unsafe { *self.noise.get_unchecked(addr % self.noise.len()) }
    }

    #[inline(always)]
    pub fn indices(&self) -> [usize; 8] {
        core::array::from_fn(|i| {
            usize::from_le_bytes([
                self.state[4 * i],
                self.state[4 * i + 1],
                self.state[4 * i + 2],
                self.state[4 * i + 3],
                0,
                0,
                0,
                0,
            ])
        })
    }
}

fn as_usize_slice(bytes: &[u8]) -> Option<&[usize]> {
    // Check if the slice is properly aligned and sized
    let align = mem::align_of::<usize>();
    let size = mem::size_of::<usize>();
    if bytes.as_ptr() as usize % align == 0 && bytes.len() % size == 0 {
        Some(unsafe { as_usize_slice_unchecked(bytes) })
    } else {
        None
    }
}

unsafe fn as_usize_slice_unchecked(bytes: &[u8]) -> &[usize] {
    let len = bytes.len() / mem::size_of::<usize>();
    std::slice::from_raw_parts(bytes.as_ptr() as *const usize, len)
}
