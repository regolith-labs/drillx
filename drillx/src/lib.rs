#[cfg(not(feature = "solana"))]
use sha3::Digest;

/// Generates a new drillx hash from a challenge and nonce.
#[inline(always)]
pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8]) -> [u8; 32] {
    let mut src: [u8; 64] = [0; 64];
    src[..32].copy_from_slice(challenge);
    src[32..40].copy_from_slice(nonce);
    src[40..48].copy_from_slice(nonce);
    src[48..56].copy_from_slice(nonce);
    src[56..64].copy_from_slice(nonce);
    haraka512_through::<6>(&src)
}

#[cfg(not(feature = "solana"))]
fn haraka512_through<const N_ROUNDS: usize>(src: &[u8; 64]) -> [u8; 32] {
    let mut dst = [0; 32];
    haraka::haraka512::<N_ROUNDS>(&mut dst, src);
    dst
}

#[cfg(feature = "solana")]
fn haraka512_through<const N_ROUNDS: usize>(src: &[u8; 64]) -> [u8; 32] {
    let mut dst = [0; 32];
    haraka_bpf::haraka512::<N_ROUNDS>(&mut dst, src);
    dst
}

/// Returns a keccak hash of the provided digest and nonce.
/// The digest is sorted prior to hashing to prevent malleability.
/// Delegates the hash to a syscall if compiled for the solana runtime.
#[cfg(feature = "solana")]
#[inline(always)]
fn keccak(msg: &[u8; 32]) -> [u8; 32] {
    solana_program::keccak::hashv(&[msg]).to_bytes()
}

/// Calculates a hash from the provided digest and nonce.
/// The digest is sorted prior to hashing to prevent malleability.
#[cfg(not(feature = "solana"))]
#[inline(always)]
fn keccak(msg: &[u8; 32]) -> [u8; 32] {
    let mut hasher = sha3::Keccak256::new();
    hasher.update(msg);
    hasher.finalize().into()
}

/// Returns the number of leading zeros on a 32 byte buffer.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haraka512_through() {
        let src = [0; 64];
        let x = haraka512_through::<6>(&src);
        println!("x: {:?}", x);
    }
}
