use std::time::Instant;

use num_enum::{IntoPrimitive, TryFromPrimitive};

/// Maximum allowed operations per digest byte
const OPCOUNT_LIMIT: u32 = 32;

/// Modulo operand for exit condition
const EXIT_OPERAND: u8 = 17;

/// Size of the digest to build
// TODO Maybe make this a variable (could be useful to tune later for onchain performance)
const DIGEST_SIZE: usize = 32;

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

    /// Timers
    t1: u128,
    t2: u128,
    t3: u128,
}

impl<'a> Operator<'a> {
    // Initialize 512 bit internal state from two recursive blake hashes
    pub fn new(challenge: &'a [u8; 32], nonce: &'a [u8; 8], noise: &'a [u8]) -> Operator<'a> {
        // Recursive blake3 hash
        let a;
        let b;
        #[cfg(not(feature = "solana"))]
        {
            a = blake3::hash(&[challenge.as_slice(), nonce.as_slice()].concat())
                .as_bytes()
                .to_owned();
            b = blake3::hash(a.as_slice()).as_bytes().to_owned();
        }

        // Recursive blake3 hash (solana runtime)
        #[cfg(feature = "solana")]
        {
            a = solana_program::blake3::hashv(&[challenge.as_slice(), nonce.as_slice()]).0;
            b = solana_program::blake3::hashv(a.as_slice()).0;
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
            t1: 0,
            t2: 0,
            t3: 0,
        }
    }

    /// Build digest using unpredictable and non-parallelizable operations
    pub fn drill(&mut self) -> [u8; DIGEST_SIZE] {
        let mut result = [0; DIGEST_SIZE];
        for i in 0..DIGEST_SIZE {
            while !self.update() {}
            result[i] = self.noise
                [usize::from_le_bytes(self.buf::<8>(self.opcount as usize)) % self.noise.len()];
            self.opcount = 0;
        }

        // Print timers
        println!("Noise {} ns", self.t1);
        println!("Op {} ns", self.t1);
        result
    }

    /// Do unpredictable number of arithmetic operations on internal state
    fn update(&mut self) -> bool {
        // Do arithmetic
        let mut b = self.state[0];
        for i in 0..64 {
            let buf = self.buf::<8>(b.wrapping_add(i) as usize);
            let n = self.noise::<64>();
            // TODO Loop an unpredictable number of times (combinations of branches must exceed what brute force can reasonably do)
            // for _ in 0..8 {
            //     //n.max(8) {
            //     n ^= dbg!(self.op(dbg!(n), dbg!(x)));
            //     x ^= dbg!(self.op(dbg!(x), dbg!(n)));
            // }
            let a = buf[self.state[i as usize % 64] as usize % 8];
            let r = buf[self.state[a as usize % 64] as usize % 8];
            self.state[i as usize] ^= self.op(a, n).rotate_right(r as u32);
            b ^= self.state[i as usize];
        }

        // Exit
        self.exit()
    }

    /// Build an unpredictable buffer of arbitrary size from a seed position and internal state
    fn buf<const N: usize>(&self, seed: usize) -> [u8; N] {
        let mut buf = [0u8; N];
        let mut a = self.state[seed % 64];
        for i in 0..N {
            buf[i] = a ^ self.state[a as usize % 64];
            a ^= buf[i].rotate_right(a.wrapping_add(i as u8) as u32 % 8);
        }
        buf
    }

    /// Read a slice of noise in a looping and unpredictable manner
    fn noise<const N: usize>(&mut self) -> u8 {
        let t = Instant::now();

        // Fill the noise buffer
        let offset = usize::from_le_bytes(self.buf::<8>(0));
        let mut addr = usize::from_le_bytes(self.buf::<8>(1));
        let mut result = self.state[0];
        for _ in 0..N {
            addr ^= usize::from_le_bytes([
                self.noise[(addr ^ offset) % self.noise.len()],
                self.noise[(addr ^ offset.rotate_right(8)) % self.noise.len()],
                self.noise[(addr ^ offset.rotate_right(16)) % self.noise.len()],
                self.noise[(addr ^ offset.rotate_right(24)) % self.noise.len()],
                self.noise[(addr ^ offset.rotate_right(32)) % self.noise.len()],
                self.noise[(addr ^ offset.rotate_right(40)) % self.noise.len()],
                self.noise[(addr ^ offset.rotate_right(48)) % self.noise.len()],
                self.noise[(addr ^ offset.rotate_right(56)) % self.noise.len()],
            ]);
            result ^= self.noise[addr % self.noise.len()];
        }

        // Shuffle and return
        self.t1 += t.elapsed().as_nanos();
        result
    }

    /// Select an opcode from the current state
    fn opcode(&mut self) -> Opcode {
        Opcode::try_from(
            (usize::from_le_bytes(self.buf::<8>(self.opcount as usize)) % Opcode::cardinality())
                as u8,
        )
        .expect("Unknown opcode")
    }

    /// Execute a random operation the given operands
    fn op(&mut self, a: u8, b: u8) -> u8 {
        let t = Instant::now();
        let x = match self.opcode() {
            Opcode::Add => a.wrapping_add(b),
            Opcode::Sub => a.wrapping_sub(b),
            Opcode::Mul => a.wrapping_mul(b),
            // TODO Avoid case where wrapping_div goes to zero if b > a
            // Opcode::Div => a.wrapping_div(b.saturating_add(2)),
            // Opcode::Nop => a,
            // Opcode::Swap => b,
        };
        self.opcount += 1;
        self.t2 += t.elapsed().as_nanos();
        x
    }

    /// Stop executing on the current byte of the digest
    fn exit(&mut self) -> bool {
        let buf = self.buf::<8>(self.opcount as usize);
        u64::from_be_bytes(buf) % EXIT_OPERAND as u64 == self.exit.into()
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
