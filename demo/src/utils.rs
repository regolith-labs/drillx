pub fn read_noise(addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> [u8; 8] {
    let mut result = [0u8; 8];
    for i in 0..16 {
        let n = noise[*addr as usize % noise.len()];
        result[i % 8] = n ^ challenge[n as usize % 32] ^ nonce[n as usize % 8];
        *addr = modpow(
            addr.saturating_add(2),
            u64::from_le_bytes(result),
            noise.len() as u64,
        );
    }
    result
}

pub fn modpow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1;
    base = base % modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % modulus;
        }
        exp = exp >> 1;
        base = base * base % modulus;
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

pub fn exit(addr: &mut u64) -> bool {
    *addr % 17 == 5
}
