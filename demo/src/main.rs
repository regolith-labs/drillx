use std::{fs::File, io::Read, ops::Add, time::Instant};

use num_bigint::BigInt;
use num_enum::TryFromPrimitive;
use num_traits::{FromBytes, ToPrimitive};
use sha3::{Digest, Keccak256};

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
    // Get some randomness
    let a = blake3::hash(&[challenge.as_ref(), nonce.to_le_bytes().as_ref()].concat());

    // Generate starting address
    let len = noise.len();
    let mut addr = modpow(
        u64::from_le_bytes(a.as_bytes()[0..8].try_into().unwrap()),
        u32::from_le_bytes(a.as_bytes()[8..12].try_into().unwrap()),
        len as u64,
    );

    // Execute random sequential lookups
    let mut digest = [0; 32];
    let mut j = noise[addr as usize];
    for i in 0..32 {
        // Select op
        let opcode = a.as_bytes()[i];
        let op = Op::try_from(opcode % OP_COUNT).expect("Unknown opcode");

        // Fetch arg
        // TODO Can skip on sqrt + nop
        let arg = get_arg(addr, j, noise);

        // Execute op
        addr = match op {
            Op::Add => addr.wrapping_add(u64::from_le_bytes(arg)),
            Op::Sub => addr.wrapping_sub(u64::from_le_bytes(arg)),
            Op::Mul => addr.wrapping_mul(u64::from_le_bytes(arg)),
            Op::Div => addr.wrapping_div(u64::from_le_bytes(arg)),
            Op::Left => addr.rotate_left(u32::from_le_bytes(arg[0..4].try_into().unwrap())),
            Op::Right => addr.rotate_right(u32::from_le_bytes(arg[0..4].try_into().unwrap())),
            Op::Xor => addr ^ u64::from_le_bytes(arg),
            Op::Sqrt => f64::from_le_bytes(addr.to_le_bytes()).sqrt() as u64,
            Op::Nop => addr,
        } % len as u64;

        // Update digest
        digest[i] = noise[addr as usize];
        j = digest[i];
    }

    // Return
    digest
}

// TODO Probably want to do ring buffer lookup for noise rather than saturating_sub(8)
fn get_arg(addr: u64, count: u8, noise: &[u8]) -> [u8; 8] {
    let mut val = [0u8; 8];
    let mut addr = addr.saturating_sub(8);
    for _ in 0..count {
        val = noise[addr as usize..(addr as usize + 8)]
            .try_into()
            .unwrap();
        // TODO Can skip last time through the loop
        addr = modpow(
            addr,
            u32::from_le_bytes(val[0..4].try_into().unwrap()),
            noise.len() as u64,
        )
        .saturating_sub(8);
    }
    val
}

fn modpow(a: u64, exp: u32, m: u64) -> u64 {
    a.wrapping_pow(exp) % m
}

pub const OP_COUNT: u8 = 9;

#[derive(Debug, PartialEq, TryFromPrimitive)]
#[repr(u8)]
pub enum Op {
    Add = 0,
    Sub,
    Mul,
    Div,
    Left,
    Right,
    Xor,
    Sqrt,
    Nop,
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
