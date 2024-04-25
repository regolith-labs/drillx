use std::{mem, time::Instant};

use num_enum::{IntoPrimitive, TryFromPrimitive};
#[cfg(not(feature = "solana"))]
use sha3::{Digest, Keccak256};

/// Digest size
// const D: usize = 16;

/// Number of recursive reads per loop
const READS: usize = 1024;

/// Number of recursive ops per loop
const OPS: usize = 512;

/// Global state for drilling algorithm
pub struct Operator2<'a> {
    state: [u8; 32],
    noise: &'a [usize],
}

impl<'a> Operator2<'a> {
    pub fn new(challenge: &'a [u8; 32], nonce: &'a [u8; 8], noise: &'a [u8]) -> Operator2<'a> {
        Operator2 {
            noise: as_usize_slice(noise).expect("Failed to read noise as &[usize]"),
            state: solana_program::keccak::hashv(&[&challenge.as_slice(), &nonce.as_slice()]).0,
        }
    }

    /// Build digest using unpredictable and non-parallelizable operations
    pub fn drill(&mut self) -> [u8; 32] {
        let mut hash_time = 0;
        let mut read_time = 0;
        let mut op_time = 0;
        let mut r = usize::from_le_bytes([
            self.state(0),
            self.state(4),
            self.state(8),
            self.state(12),
            self.state(16),
            self.state(20),
            self.state(24),
            self.state(28),
        ]);
        for i in 0..4 {
            // Fetch noise
            let read_timer = Instant::now();
            let idx = self.indices();
            for j in 0..8 {
                self.state[8 * i + j] ^= self.do_reads::<READS>(idx[j], r);
            }
            read_time += read_timer.elapsed().as_nanos();

            // Do ops
            let op_timer = Instant::now();
            let idx = self.indices();
            for j in 0..OPS {
                r ^= self.op(idx[j % 8], r, j);
            }
            op_time += op_timer.elapsed().as_nanos();

            // Hash state
            let hash_timer = Instant::now();
            self.hash(r);
            hash_time += hash_timer.elapsed().as_nanos();
        }

        println!(
            "op {op_time} hash {hash_time} vs read {read_time}: {}x {}x",
            read_time as f64 / hash_time as f64,
            read_time as f64 / op_time as f64
        );

        self.state
    }

    #[inline(always)]
    fn hash(&mut self, r: usize) {
        self.state = solana_program::keccak::hashv(&[&self.state, r.to_le_bytes().as_slice()]).0;
    }

    #[inline(always)]
    fn do_reads<const R: usize>(&mut self, mut index: usize, r: usize) -> u8 {
        for _ in 0..R {
            index ^= self.noise(index);
        }
        (self.noise(index) >> (r % 8)) as u8
    }

    #[inline(always)]
    fn noise(&self, addr: usize) -> usize {
        unsafe { *self.noise.get_unchecked(addr % self.noise.len()) }
    }

    #[inline(always)]
    fn state(&self, i: usize) -> u8 {
        unsafe { *self.state.get_unchecked(i % 32) }
    }

    #[inline(always)]
    fn indices(&self) -> [usize; 8] {
        core::array::from_fn(|i| {
            usize::from_le_bytes([
                self.state(4 * i),
                self.state(4 * i + 1),
                self.state(4 * i + 2),
                self.state(4 * i + 3),
                0,
                0,
                0,
                0,
            ])
        })
    }

    /// Select an opcode from the current state
    #[inline]
    fn opcode(&self, opcount: usize, b: usize) -> Opcode {
        Opcode::try_from(((opcount ^ b) % Opcode::cardinality()) as u8).expect("Unknown opcode")
    }

    /// Execute a random operation the given operands
    #[inline]
    fn op(&self, a: usize, b: usize, opcount: usize) -> usize {
        match self.opcode(opcount, b) {
            Opcode::Add => a.wrapping_add(b),
            Opcode::Sub => a.wrapping_sub(b),
            Opcode::Mul => a.wrapping_mul(b),
            Opcode::Div => {
                if a.gt(&b) {
                    a.wrapping_div(b.saturating_add(2))
                } else {
                    b.wrapping_div(a.saturating_add(2))
                }
            }
            Opcode::Xor => a ^ b,
            Opcode::Right => a >> (b % 64),
            Opcode::Left => a << (b % 64),
        }
    }
}

/// Set of arbitrary compute operations to chose from
#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
enum Opcode {
    Add = 0,
    Sub,
    Mul,
    Div,
    Xor,
    Right,
    Left,
}

impl Opcode {
    pub fn cardinality() -> usize {
        7
    }
}

// Check if the slice is properly aligned and sized
fn as_usize_slice(bytes: &[u8]) -> Option<&[usize]> {
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
