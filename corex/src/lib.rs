mod siphash;

use siphash::{siphash24_ctr, SipState};

/// Pre-built hash program that can be rapidly computed with different inputs
///
/// The program and initial state representation are not specified in this
/// public interface, but [`std::fmt::Debug`] can describe program internals.
#[derive(Debug)]
pub struct CoreX {
    sipstate: SipState,
}

impl CoreX {
    /// The maximum available output size for [`Self::hash_to_bytes()`]
    pub const FULL_SIZE: usize = 32;

    /// Generate a new hash function with the supplied seed.
    pub fn new(input: &[u8; 64]) -> Self {
        use blake2::{Blake2s256, Digest};
        let mut hasher = Blake2s256::new();
        hasher.update(input);
        let seed: [u8; 32] = hasher.finalize().try_into().unwrap();
        Self {
            // challenge: *challenge,
            sipstate: SipState::new_from_bytes(&seed),
        }
    }

    /// Calculate the first 64-bit word of the hash, without converting to bytes.
    pub fn hash_to_u64(&self, input: u64) -> u64 {
        let hash = self.hash_to_bytes(input);
        let s0 = hash[0..8].try_into().unwrap();
        let s1 = hash[8..16].try_into().unwrap();
        let s2 = hash[16..24].try_into().unwrap();
        let s3 = hash[24..32].try_into().unwrap();
        u64::from_le_bytes(s0)
            ^ u64::from_le_bytes(s1)
            ^ u64::from_le_bytes(s2)
            ^ u64::from_le_bytes(s3)
    }

    /// Calculate the hash function at its full output width, returning a fixed
    /// size byte array.
    pub fn hash_to_bytes(&self, input: u64) -> [u8; Self::FULL_SIZE] {
        let sip = siphash24_ctr(self.sipstate, input);
        let seed: [u8; 64] = unsafe { std::mem::transmute(sip) };

        #[cfg(any(feature = "cpu", feature = "cpu-bpf"))]
        let hash = haraka512_through::<6>(&seed);

        #[cfg(any(feature = "gpu", feature = "gpu-bpf"))]
        let hash = {
            let mut hasher = Blake2s256::new();
            hasher.update(&seed);
            hasher.finalize().try_into().unwrap()
        };

        hash
    }
}

fn haraka512_through<const N_ROUNDS: usize>(src: &[u8; 64]) -> [u8; 32] {
    let mut dst = [0; 32];

    #[cfg(feature = "cpu")]
    haraka::haraka512::<N_ROUNDS>(&mut dst, src);

    #[cfg(feature = "cpu-bpf")]
    haraka_bpf::haraka512::<N_ROUNDS>(&mut dst, src);

    dst
}
