/// Re-export the equix crate
pub use equix;

/// Generates a new drillx hash from a challenge and nonce
pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<Hash, DrillxError> {
    let digest = build_digest(challenge, nonce)?;
    Ok(Hash {
        d: digest,
        h: hashv(&digest, nonce),
    })
}

/// Generates a new drillx hash from a challenge and nonce
#[inline(always)]
pub fn hash_with_shared_memory(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Result<Hash, DrillxError> {
    // Seed
    let seed = construct_seed(challenge, nonce);
    let digest = build_digest_with_shared_memory(memory, &seed)?;
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

/// Concatenates two arrays into a single array
#[inline(always)]
fn construct_seed(a: &[u8; 32], b: &[u8; 8]) -> [u8; 40] {
    let mut result = std::mem::MaybeUninit::uninit();
    let dest = result.as_mut_ptr() as *mut u8;
    // SAFETY: `dest` is valid for `40` elements.
    // SAFETY: `a` and `b` are valid for `32` and `8` elements respectively.
    // SAFETY: `a` and `b` are non-overlapping with `dest`.
    unsafe {
        core::ptr::copy_nonoverlapping(a.as_ptr(), dest, 32);
        core::ptr::copy_nonoverlapping(b.as_ptr(), dest.add(32), 8);
        result.assume_init()
    }
}

/// Constructs a blake3 digest from a challenge and nonce using equix hashes
fn build_digest(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<[u8; 16], DrillxError> {
    let seed = seed(challenge, nonce);
    let Ok(solutions) = equix::solve(&seed) else {
        return Err(DrillxError::BadEquix);
    };
    let Some(solution) = solutions.first() else {
        return Err(DrillxError::BadEquix);
    };

    // Digest
    Ok(solution.to_bytes())
}

/// Constructs a blake3 digest from a challenge and nonce using equix hashes
#[inline(always)]
fn build_digest_with_shared_memory(
    memory: &mut equix::SolverMemory,
    seed: &[u8],
) -> Result<[u8; 16], DrillxError> {
    let equix = equix::EquiXBuilder::new()
        .runtime(equix::RuntimeOption::CompileOnly)
        .build(&seed)
        .map_err(|_| DrillxError::BadEquix)?;

    // Equix
    let solutions = equix.solve_with_memory(memory);
    // SAFETY: The equix solver guarantees that the first solution is always valid
    let solution = unsafe { solutions.get_unchecked(0) };

    // Digest
    Ok(solution.to_bytes())
}

/// Returns true if the digest if valid equihash construction from the challenge and nonce
pub fn is_valid_digest(challenge: &[u8; 32], nonce: &[u8; 8], digest: &[u8; 16]) -> bool {
    let seed = seed(challenge, nonce);
    equix::verify_bytes(&seed, digest).is_ok()
}

/// Calculates a hash from the provided digest and nonce.
/// The digest is sorted prior to hashing to prevent malleability.
#[cfg(all(feature = "program", not(feature = "native")))]
fn hashv(digest: &[u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    unsafe {
        let mut u16_slice: [u16; 8] = std::mem::transmute_copy(digest);
        u16_slice.sort_unstable();
        let u8_slice: [u8; 16] = std::mem::transmute(u16_slice);
        solana_program::blake3::hashv(&[u8_slice.as_slice(), &nonce.as_slice()]).to_bytes()
    }
}

/// Calculates a hash from the provided digest and nonce
/// The digest is sorted prior to hashing to prevent malleability.
#[cfg(all(feature = "native", not(feature = "program")))]
fn hashv(digest: &mut [u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    let u8_slice: &mut [u8; 16] = unsafe {
        let u16_slice: &mut [u16; 8] = core::mem::transmute(digest);
        u16_slice.sort_unstable();
        core::mem::transmute(u16_slice);
    };
    // Hash an input incrementally.
    let mut hasher = blake3::Hasher::new();
    hasher.update(u8_slice);
    hasher.update(nonce);
    hasher.finalize().into()
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
