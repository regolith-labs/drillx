pub fn read_noise(addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> [u8; 8] {
    let mut result = [0u8; 8];
    for i in 0..16 {
        let n = noise[*addr as usize % noise.len()];
        result[i % 8] = n ^ challenge[n as usize % 32] ^ nonce[n as usize % 8];
        *addr = modpow(
            *addr,
            u64::from_le_bytes([result[i % 8], result[(i + 1) % 8], 0, 0, 0, 0, 0, 0]) as u64,
            u64::MAX / 256,
        );
    }
    result
}

pub fn modpow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let startb = base;
    let starte = exp;
    base = base.saturating_add(2);
    let mut result = 1u64;
    base = base % modulus; // Take initial modulo to reduce the size.
    while exp > 0 {
        if exp % 2 == 1 {
            result = result.wrapping_mul(base) % modulus;
        }
        exp >>= 1; // Right shift exp by 1
        base = base.wrapping_mul(base) % modulus; // Square the base and take modulo
    }
    if result.eq(&0) {
        panic!("Modpow is zero: {} {} {}", startb, starte, modulus);
    }
    result
}

pub fn difficulty(hash: [u8; 32]) -> u32 {
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

// TODO Make exit condition dynamic (*addr % 17 = f(challenge, nonce))
pub fn exit(addr: &mut u64) -> bool {
    *addr % 17 == 5
}
