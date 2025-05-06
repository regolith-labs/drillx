#[derive(Debug)]
pub struct VerusX {
    challenge: [u8; 64],
}

impl VerusX {
    /// The maximum available output size for [`Self::hash_to_bytes()`]
    pub const FULL_SIZE: usize = 32;

    /// Generate a new hash function with the supplied seed.
    pub fn new(challenge: &[u8; 64]) -> Self {
        Self {
            challenge: *challenge,
        }
    }

    /// Calculate the first 64-bit word of the hash, without converting to bytes.
    pub fn hash_to_u64(&self, input: u64) -> u64 {
        let hash = self.hash_to_bytes(input);
        u64::from_le_bytes(hash[0..8].try_into().unwrap())
    }

    /// Calculate the hash function at its full output width, returning a fixed
    /// size byte array.
    pub fn hash_to_bytes(&self, input: u64) -> [u8; Self::FULL_SIZE] {
        let mut src = [0; 72];
        // src[0..8].copy_from_slice(&input.to_le_bytes());
        // src[8..72].copy_from_slice(&self.challenge);
        // src[0..64].copy_from_slice(&self.challenge);
        // src[64..72].copy_from_slice(&input.to_le_bytes());
        let mut src = [0; 96];

        let solution = verus_rs::VerusHashV2::hash(&src);
        solution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verus_hash() {
        let challenge = [0; 64];
        let verus = VerusX::new(&challenge);
        let solution = verus.hash_to_bytes(0u64);
        let solution2 = verus.hash_to_bytes(1u64);
        println!("solution: {:?}", solution);
        println!("solution2: {:?}", solution2);
    }
}
