use num_enum::{IntoPrimitive, TryFromPrimitive};

// TODO Maybe make all consts into variables (could be useful to tune later for onchain performance)

/// Modulo operand for exit condition
const EXIT_OPERAND: u8 = 7;

/// Size of the digest to build
/// This needs to be at least 8 (bare minimum) to avoid collisions.
/// The challenge is provided to the user. So the only user input here is the u64 nonce.
/// If the digest size is less than 8 bytes, drill is guranteed to produce collisions for different
/// nonce values given the same challenge.
const DIGEST_SIZE: usize = 1;

/// How many loops to do per noise read
const READ_HEAVINESS: u64 = 0;

/// Global state for drilling algorithm
#[derive(Debug)]
pub struct Operator<'a> {
    /// Scratchpad for bit noise
    noise: &'a [u8],

    /// 512-bit internal state
    state: [u8; 64],

    /// Operations executed on current slice of the digest
    opcount: u32,

    /// Exit condition
    exit: u8,
}

impl<'a> Operator<'a> {
    // Initialize 512 bit internal state from two recursive blake hashes
    pub fn new(challenge: &'a [u8; 32], nonce: &'a [u8; 8], noise: &'a [u8]) -> Operator<'a> {
        // Recursive blake3 hash
        let a;
        let b;
        #[cfg(not(feature = "solana"))]
        {
            a = blake3::hash(&[challenge.as_slice(), nonce.as_slice()].concat()).as_bytes();
            b = blake3::hash(&[a.as_slice()]).as_bytes();
        }

        // Recursive blake3 hash (solana runtime)
        #[cfg(feature = "solana")]
        {
            a = solana_program::blake3::hashv(&[challenge.as_slice(), nonce.as_slice()]).0;
            b = solana_program::blake3::hashv(&[a.as_slice()]).0;
        }

        // Build internal state
        let mut state = [0u8; 64];
        for i in 0..32 {
            state[i * 2] = a[i];
            state[i * 2 + 1] = b[i];
        }

        Operator {
            noise,
            state,
            exit: state[0] % EXIT_OPERAND,
            opcount: 0,
        }
    }

    /// Build digest using unpredictable and non-parallelizable operations
    pub fn drill(&mut self) -> [u8; DIGEST_SIZE] {
        [0; DIGEST_SIZE]
        // let mut r = self.state[0];
        // let mut result = [0; DIGEST_SIZE];
        // for i in 0..DIGEST_SIZE {
        //     while !self.update() {}
        //     let addr = usize::from_le_bytes(self.buf(self.opcount as u8));
        //     let n = self.noise[addr % self.noise.len()];
        //     r = r.wrapping_add(n);
        //     result[i] = n.rotate_right(r as u32 % 8);
        //     self.opcount = 0;
        // }
        // result
    }

    /// Do unpredictable number of arithmetic operations on internal state
    // TODO Loop op an unpredictable number of times.
    //      Probability combinations of op branches must exceed what brute force can reasonably do.
    fn update(&mut self) -> bool {
        // Do arithmetic
        let mut b = self.state[0];
        for i in 0..64 {
            let n = self.noise();
            // let a = self.state[n.wrapping_add(b) as usize % 64];
            // self.state[i as usize] ^= self.op(a, n);
            self.state[i as usize] ^= self.op(n, b);
            b ^= self.state[i as usize];
        }

        // Exit
        self.exit(b)
    }

    /// Build a buffer of arbitrary size from a seed and internal state
    // TODO Optimize buf calls.
    //      Buf itself is pretty cheap, but we call it twice as often as op.
    fn buf(&mut self, seed: u8) -> [u8; 8] {
        [
            self.state[seed as usize % 64],
            self.state[seed.rotate_right(1) as usize % 64],
            self.state[seed.rotate_right(2) as usize % 64],
            self.state[seed.rotate_right(3) as usize % 64],
            self.state[seed.rotate_right(4) as usize % 64],
            self.state[seed.rotate_right(5) as usize % 64],
            self.state[seed.rotate_right(6) as usize % 64],
            self.state[seed.rotate_right(7) as usize % 64],
        ]
    }

    /// Read a slice of noise in a looping and unpredictable manner
    // TODO Do we need 2 buf calls?
    // TODO Test alternative method of addr construction that doesn't rely on hardcoded rotations
    fn noise(&mut self) -> u8 {
        // Fill the noise buffer
        let mut result = self.state[0]; //self.sum();
        let mask = usize::from_le_bytes(self.buf(result));
        let mut addr = usize::from_le_bytes(self.buf(mask as u8));
        for _ in 0..READ_HEAVINESS {
            addr ^= usize::from_le_bytes([
                self.noise[(addr ^ mask) % self.noise.len()],
                self.noise[(addr ^ mask.rotate_right(8)) % self.noise.len()],
                self.noise[(addr ^ mask.rotate_right(16)) % self.noise.len()],
                self.noise[(addr ^ mask.rotate_right(24)) % self.noise.len()],
                self.noise[(addr ^ mask.rotate_right(32)) % self.noise.len()],
                self.noise[(addr ^ mask.rotate_right(40)) % self.noise.len()],
                self.noise[(addr ^ mask.rotate_right(48)) % self.noise.len()],
                self.noise[(addr ^ mask.rotate_right(56)) % self.noise.len()],
            ]);
            result ^= self.noise[addr % self.noise.len()];
        }

        // Return
        result
    }

    /// Select an opcode from the current state
    fn opcode(&mut self) -> Opcode {
        Opcode::try_from(self.state[self.opcount as usize % 64] % Opcode::cardinality() as u8)
            .expect("Unknown opcode")
    }

    /// Execute a random operation the given operands
    // TODO Avoid case where wrapping_div goes to zero if b > a
    fn op(&mut self, a: u8, b: u8) -> u8 {
        let x = match self.opcode() {
            Opcode::Add => a.wrapping_add(b),
            Opcode::Sub => a.wrapping_sub(b),
            Opcode::Mul => a.wrapping_mul(b),
            // Opcode::Div => a.wrapping_div(b.saturating_add(2)),
            // Opcode::Nop => a,
            // Opcode::Swap => b,
        };
        self.opcount += 1;
        x
    }

    /// Stop executing on the current byte of the digest
    fn exit(&mut self, seed: u8) -> bool {
        seed % EXIT_OPERAND == self.exit.into()
    }
}

/// Set of arbitrary compute operations to chose from
// TODO Add more ops
// TODO Div, Sqrt, Floats ?
#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
enum Opcode {
    Add = 0,
    Sub,
    Mul,
    // Div,
    // Nop,
    // Swap,
}

impl Opcode {
    pub fn cardinality() -> usize {
        3 // 6
    }
}
