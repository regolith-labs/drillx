pub use equix;

/// Generates a new drillx hash from a challenge and nonce
pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<Hash, DrillxError> {
    let mut digest = digest(challenge, nonce)?;
    Ok(Hash {
        d: digest,
        h: hashv(&mut digest, nonce),
    })
}

/// Generates a new drillx hash from a challenge and nonce using shared memory
#[inline(always)]
pub fn hash_with_shared_memory(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Result<Hash, DrillxError> {
    let mut digest = digest_with_shared_memory(challenge, nonce, memory)?;
    Ok(Hash {
        d: digest,
        h: hashv(&mut digest, nonce),
    })
}

/// Concatenates a challenge and a nonce into a single buffer
#[inline(always)]
fn seed(a: &[u8; 32], b: &[u8; 8]) -> [u8; 40] {
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
fn digest(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<[u8; 16], DrillxError> {
    let seed = seed(challenge, nonce);
    let solutions = equix::solve(&seed).map_err(|_| DrillxError::BadEquix)?;
    // SAFETY: The equix solver guarantees that the first solution is always valid
    let solution = unsafe { solutions.get_unchecked(0) };
    Ok(solution.to_bytes())
}

/// Constructs a blake3 digest from a challenge and nonce using equix hashes
#[inline(always)]
fn digest_with_shared_memory(
    challenge: &[u8; 32],
    nonce: &[u8; 8],
    memory: &mut equix::SolverMemory,
) -> Result<[u8; 16], DrillxError> {
    let seed = seed(challenge, nonce);
    let equix = equix::EquiXBuilder::new()
        .runtime(equix::RuntimeOption::CompileOnly)
        .build(&seed)
        .map_err(|_| DrillxError::BadEquix)?;
    let solutions = equix.solve_with_memory(memory);
    // SAFETY: The equix solver guarantees that the first solution is always valid
    let solution = unsafe { solutions.get_unchecked(0) };
    Ok(solution.to_bytes())
}

/// Returns true if the digest if valid equihash construction from the challenge and nonce
pub fn is_valid_digest(challenge: &[u8; 32], nonce: &[u8; 8], digest: &[u8; 16]) -> bool {
    let seed = seed(challenge, nonce);
    equix::verify_bytes(&seed, digest).is_ok()
}

/// Sorts the provided digest as a list of u16 values.
fn sorted(digest: &mut [u8; 16]) -> &mut [u8; 16] {
    unsafe {
        let u16_slice: &mut [u16; 8] = core::mem::transmute(digest);
        u16_slice.sort_unstable();
        core::mem::transmute(u16_slice)
    }
}

/// Returns a blake3 hash of the provided digest and nonce.
/// The digest is sorted prior to hashing to prevent malleability.
/// Delegates the hash to a syscall if compiled for the solana runtime.
#[cfg(feature = "solana")]
fn hashv(digest: &mut [u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    solana_program::blake3::hashv(&[sorted(digest).as_slice(), &nonce.as_slice()]).to_bytes()
}

/// Calculates a hash from the provided digest and nonce.
/// The digest is sorted prior to hashing to prevent malleability.
#[cfg(not(feature = "solana"))]
fn hashv(digest: &mut [u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(sorted(digest));
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
