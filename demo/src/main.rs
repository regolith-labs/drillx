use std::{fs::File, io::Read, time::Instant};

use enum_dispatch::enum_dispatch;
use sha3::{Digest, Keccak256};
use strum::IntoEnumIterator;

const TARGET_DIFFICULTY: u32 = 4; //10;

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

    // The drill part (1024 sequential modpow and mem reads)
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
    let len = noise.len();
    let r = blake3::hash(&[challenge.as_ref(), nonce.to_le_bytes().as_ref()].concat());
    let mut addr = modpow(
        u64::from_le_bytes(r.as_bytes()[0..8].try_into().unwrap()),
        u32::from_le_bytes(r.as_bytes()[8..12].try_into().unwrap()),
        len as u64,
    );

    // Build digest
    let nonce_ = nonce.to_le_bytes();
    let mut digest = [0; 32];
    for i in 0..32 {
        // Do random ops until exit
        while !random_op(ops, &mut addr, &challenge, &nonce_, noise)
            .op(&mut addr, &challenge, &nonce_)
        {
            // Noop
        }

        // Append to digest
        digest[i] = get_val(addr, r.as_bytes()[i as usize % 32], noise)[i % 8];
    }

    // Return
    digest
}

// TODO Probably want to do ring buffer lookup for noise rather than saturating_sub(8)
fn get_val(addr: u64, r: u8, noise: &[u8]) -> [u8; 8] {
    let mut r = r;
    let mut addr = (addr % noise.len() as u64).saturating_sub(8);
    loop {
        let val: [u8; 8] = noise[addr as usize..(addr as usize + 8)]
            .try_into()
            .unwrap();
        // TODO Can skip last time through the loop
        addr = modpow(
            addr,
            u32::from_le_bytes(val[0..4].try_into().unwrap()),
            noise.len() as u64,
        )
        .saturating_sub(8);

        r = r.wrapping_add(1); // TODO This is fishy
        if r % 17 == 5 {
            return val;
        }
    }
}

fn modpow(a: u64, exp: u32, m: u64) -> u64 {
    a.wrapping_pow(exp) % m
}

fn random_op<'a>(
    ops: &'a mut [RandomOp],
    addr: &mut u64,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
    noise: &[u8],
) -> &'a mut RandomOp {
    // Seed op from challenge
    let len = noise.len();
    let mut n = noise[*addr as usize % len];
    let mut seed = [0u8; 8];
    for i in 0..8 {
        // Read A
        let a = challenge[n as usize % 32];
        *addr = modpow(*addr, a as u32, len as u64);
        n = noise[*addr as usize];

        // Read B
        let b = challenge[n as usize % 32];
        *addr = modpow(*addr, b as u32, len as u64);
        n = noise[*addr as usize];

        // Read B
        let c = challenge[n as usize % 32];
        *addr = modpow(*addr, c as u32, len as u64);
        n = noise[*addr as usize];

        // Read D
        let d = challenge[n as usize % 32];
        *addr = modpow(*addr, d as u32, len as u64);
        n = noise[*addr as usize];

        // Generate seed
        seed[i] = a ^ b ^ c ^ d;
    }
    let mut chosen_op = usize::from_le_bytes(seed);

    // Mask with nonce
    for i in 0..8 {
        seed[i] = nonce[n as usize % 8];
        // TODO Can skip on last iteration of loop
        *addr = modpow(*addr, seed[i] as u32, len as u64);
        n = noise[*addr as usize];
    }
    chosen_op ^= usize::from_le_bytes(seed);

    // Return op
    &mut ops[chosen_op % ops.len()]
}

#[derive(strum::EnumIter)]
#[enum_dispatch(Op)]
pub enum RandomOp {
    MaskAdd(MaskAdd),
    Xor(Xor),
}

pub struct MaskAdd {
    mask: u64,
    b: u64,
}

impl Default for MaskAdd {
    fn default() -> MaskAdd {
        MaskAdd {
            mask: 0b_10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010,
            b: u64::from_le_bytes(core::array::from_fn(|i| b"ore"[i % 3])),
        }
    }
}

impl Op for MaskAdd {
    fn op(&mut self, a: &mut u64, challenge: &[u8; 32], nonce: &[u8; 8]) -> bool {
        // Mix
        let a_ = a.to_le_bytes();
        self.mask ^= u64::from_le_bytes([
            a_[6] ^ challenge[a_[1] as usize % 32] ^ nonce[a_[2] as usize % 8],
            a_[1] ^ challenge[a_[2] as usize % 32] ^ nonce[a_[3] as usize % 8],
            a_[2] ^ challenge[a_[3] as usize % 32] ^ nonce[a_[0] as usize % 8],
            a_[3] ^ challenge[a_[0] as usize % 32] ^ nonce[a_[4] as usize % 8],
            a_[0] ^ challenge[a_[4] as usize % 32] ^ nonce[a_[7] as usize % 8],
            a_[4] ^ challenge[a_[7] as usize % 32] ^ nonce[a_[5] as usize % 8],
            a_[7] ^ challenge[a_[5] as usize % 32] ^ nonce[a_[6] as usize % 8],
            a_[5] ^ challenge[a_[6] as usize % 32] ^ nonce[a_[1] as usize % 8],
        ]);
        self.b ^= u64::from_le_bytes([
            a_[2] ^ challenge[a_[3] as usize % 32] ^ nonce[a_[0] as usize % 8],
            a_[0] ^ challenge[a_[4] as usize % 32] ^ nonce[a_[7] as usize % 8],
            a_[1] ^ challenge[a_[2] as usize % 32] ^ nonce[a_[3] as usize % 8],
            a_[3] ^ challenge[a_[0] as usize % 32] ^ nonce[a_[4] as usize % 8],
            a_[7] ^ challenge[a_[5] as usize % 32] ^ nonce[a_[6] as usize % 8],
            a_[6] ^ challenge[a_[1] as usize % 32] ^ nonce[a_[2] as usize % 8],
            a_[5] ^ challenge[a_[6] as usize % 32] ^ nonce[a_[1] as usize % 8],
            a_[4] ^ challenge[a_[7] as usize % 32] ^ nonce[a_[5] as usize % 8],
        ]);

        // Apply mask and add
        *a = (*a & self.mask).wrapping_add(self.b);

        // Exit code
        *a % 17 == 5
    }
}

pub struct Xor {
    mask: u64,
}

impl Default for Xor {
    fn default() -> Xor {
        Xor {
            mask: 0b_01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101,
        }
    }
}

impl Op for Xor {
    fn op(&mut self, a: &mut u64, challenge: &[u8; 32], nonce: &[u8; 8]) -> bool {
        // Mix (TODO, maybe mix these up)
        let a_ = a.to_le_bytes();
        self.mask ^= u64::from_le_bytes([
            a_[1] ^ challenge[a_[2] as usize % 32] ^ nonce[a_[3] as usize % 8],
            a_[3] ^ challenge[a_[0] as usize % 32] ^ nonce[a_[4] as usize % 8],
            a_[2] ^ challenge[a_[3] as usize % 32] ^ nonce[a_[0] as usize % 8],
            a_[5] ^ challenge[a_[6] as usize % 32] ^ nonce[a_[1] as usize % 8],
            a_[4] ^ challenge[a_[7] as usize % 32] ^ nonce[a_[5] as usize % 8],
            a_[7] ^ challenge[a_[5] as usize % 32] ^ nonce[a_[6] as usize % 8],
            a_[6] ^ challenge[a_[1] as usize % 32] ^ nonce[a_[2] as usize % 8],
            a_[0] ^ challenge[a_[4] as usize % 32] ^ nonce[a_[7] as usize % 8],
        ]);

        // Apply mask
        *a ^= self.mask;

        // Exit
        *a % 17 == 7
    }
}

#[enum_dispatch]
pub trait Op {
    fn op(&mut self, a: &mut u64, challenge: &[u8; 32], nonce: &[u8; 8]) -> bool;
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
