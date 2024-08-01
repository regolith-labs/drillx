pub use equix;
#[cfg(not(feature = "solana"))]
use sha3::Digest;

/// Generates a new drillx hash from a challenge and nonce.
#[inline(always)]
pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<Hash, DrillxError> {
    let digest = digest(challenge, nonce)?;
    Ok(Hash {
        d: digest,
        h: hashv(&digest, nonce),
    })
}

/// Generates a new drillx hash from a challenge and nonce using pre-allocated memory.
#[inline(always)]
pub fn hash_with_memory(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Result<Hash, DrillxError> {
    let digest = digest_with_memory(memory, challenge, nonce)?;
    Ok(Hash {
        d: digest,
        h: hashv(&digest, nonce),
    })
}

/// Concatenates a challenge and a nonce into a single buffer.
#[inline(always)]
pub fn seed(challenge: &[u8; 32], nonce: &[u8; 8]) -> [u8; 40] {
    let mut result = [0; 40];
    result[00..32].copy_from_slice(challenge);
    result[32..40].copy_from_slice(nonce);
    result
}

/// Constructs a keccak digest from a challenge and nonce using equix hashes.
#[inline(always)]
fn digest(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<[u8; 16], DrillxError> {
    let seed = seed(challenge, nonce);
    let solutions = equix::solve(&seed).map_err(|_| DrillxError::BadEquix)?;
    if solutions.is_empty() {
        return Err(DrillxError::NoSolutions);
    }
    // SAFETY: The equix solver guarantees that the first solution is always valid
    let solution = unsafe { solutions.get_unchecked(0) };
    Ok(solution.to_bytes())
}

/// Constructs a keccak digest from a challenge and nonce using equix hashes and pre-allocated memory.
#[inline(always)]
fn digest_with_memory(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Result<[u8; 16], DrillxError> {
    let seed = seed(challenge, nonce);
    let equix = equix::EquiXBuilder::new()
        .runtime(equix::RuntimeOption::TryCompile)
        .build(&seed)
        .map_err(|_| DrillxError::BadEquix)?;
    let solutions = equix.solve_with_memory(memory);
    if solutions.is_empty() {
        return Err(DrillxError::NoSolutions);
    }
    // SAFETY: The equix solver guarantees that the first solution is always valid
    let solution = unsafe { solutions.get_unchecked(0) };
    Ok(solution.to_bytes())
}

/// Sorts the provided digest as a list of u16 values.
#[inline(always)]
fn sorted(mut digest: [u8; 16]) -> [u8; 16] {
    unsafe {
        let u16_slice: &mut [u16; 8] = core::mem::transmute(&mut digest);
        u16_slice.sort_unstable();
        digest
    }
}

/// Returns a keccak hash of the provided digest and nonce.
/// The digest is sorted prior to hashing to prevent malleability.
/// Delegates the hash to a syscall if compiled for the solana runtime.
#[cfg(feature = "solana")]
#[inline(always)]
fn hashv(digest: &[u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    solana_program::keccak::hashv(&[sorted(*digest).as_slice(), &nonce.as_slice()]).to_bytes()
}

/// Calculates a hash from the provided digest and nonce.
/// The digest is sorted prior to hashing to prevent malleability.
#[cfg(not(feature = "solana"))]
#[inline(always)]
fn hashv(digest: &[u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    let mut hasher = sha3::Keccak256::new();
    hasher.update(&sorted(*digest));
    hasher.update(nonce);
    hasher.finalize().into()
}

/// Returns true if the digest if valid equihash construction from the challenge and nonce.
pub fn is_valid_digest(challenge: &[u8; 32], nonce: &[u8; 8], digest: &[u8; 16]) -> bool {
    let seed = seed(challenge, nonce);
    equix::verify_bytes(&seed, digest).is_ok()
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

/// The result of a drillx hash
#[derive(Default)]
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
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
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
        let mut d = self.d;
        Hash {
            d: self.d,
            h: hashv(&mut d, &self.n),
        }
    }
}

#[derive(Debug)]
pub enum DrillxError {
    BadEquix,
    NoSolutions,
}

impl std::fmt::Display for DrillxError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            DrillxError::BadEquix => write!(f, "Failed equix"),
            DrillxError::NoSolutions => write!(f, "No solutions"),
        }
    }
}

impl std::error::Error for DrillxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
