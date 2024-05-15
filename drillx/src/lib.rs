/// Generates a new drillx hash from a challenge and nonce
pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<Hash, DrillxError> {
    let digest = build_digest(challenge, nonce)?;
    Ok(Hash {
        d: digest,
        h: hashv(&digest, nonce),
    })
}

/// Generates a seed from the given challenge and nonce
fn seed(challenge: &[u8; 32], nonce: &[u8; 8]) -> [u8; 40] {
    let mut buf = [0; 40];
    buf[0..32].copy_from_slice(&challenge[..]);
    buf[32..40].copy_from_slice(&nonce[..]);
    buf
}

/// Constructs a keccak digest from a challenge and nonce using equix hashes
fn build_digest(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<[u8; 16], DrillxError> {
    // Seed
    let seed = seed(challenge, nonce);

    // Equix
    let Ok(solutions) = equix::solve(&seed) else {
        return Err(DrillxError::BadEquix);
    };
    let Some(solution) = solutions.first() else {
        return Err(DrillxError::BadEquix);
    };

    // Digest
    let mut digest = [0u8; 16];
    digest.copy_from_slice(solution.to_bytes().as_slice());
    Ok(digest)
}

/// Returns true if the digest if valid equihash construction from the challenge and nonce
pub fn is_valid_digest(challenge: &[u8; 32], nonce: &[u8; 8], digest: &[u8; 16]) -> bool {
    let seed = seed(challenge, nonce);
    equix::verify_bytes(&seed, digest).is_ok()
}

/// Calculates a hash from the provided digest and nonce
fn hashv(digest: &[u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    solana_program::blake3::hashv(&[&digest.as_slice(), &nonce.as_slice()]).to_bytes()
}

/// Returns the number of leading zeros on a 32 byte buffer
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

/// The result of a drillx hash
pub struct Hash {
    pub d: [u8; 16], // digest
    pub h: [u8; 32], // hash
}

impl Hash {
    /// The leading number of zeros on the hash
    pub fn difficulty(&self) -> u32 {
        difficulty(self.h)
    }
}

/// A drillx solution which can be efficiently validated on-chain
pub struct Solution {
    pub d: [u8; 16], // digest
    pub n: [u8; 8],  // nonce
}

impl Solution {
    /// Builds a new verifiable solution from a hash and nonce
    pub fn new(digest: [u8; 16], nonce: [u8; 8]) -> Solution {
        Solution {
            d: digest,
            n: nonce,
        }
    }

    /// Returns true if the solution is valid
    pub fn is_valid(&self, challenge: &[u8; 32]) -> bool {
        is_valid_digest(challenge, &self.n, &self.d)
    }

    /// Calculates the result hash for a given solution
    pub fn to_hash(&self) -> Hash {
        Hash {
            d: self.d,
            h: hashv(&self.d, &self.n),
        }
    }
}

#[derive(Debug)]
pub enum DrillxError {
    BadEquix,
}

impl std::fmt::Display for DrillxError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            DrillxError::BadEquix => write!(f, "Failed equix"),
        }
    }
}

impl std::error::Error for DrillxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
