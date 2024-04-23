use std::time::Instant;

use num_enum::{IntoPrimitive, TryFromPrimitive};

/// Maximum allowed operations per digest byte
const OPCOUNT_LIMIT: u32 = 32;

/// A prime number for the shuffle operation
const SHUFFLE_PRIME: u8 = 31;

/// Modulo operand for exit condition
const EXIT_OPERAND: u8 = 17;

/// Size of the digest to build
// TODO Maybe make this a variable (could be useful to tune later for onchain performance)
const DIGEST_SIZE: usize = 32;

/// Global state for drill operations
#[derive(Debug)]
pub struct Operator<'a> {
    /// Scratchpad for bit noise
    noise: &'a [u8],

    /// 512-bit internal state
    state: [u8; 64],

    /// Count of operations executed on current byte of the digest
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
        #[cfg(not(feature = "solana"))]
        let a = blake3::hash(&[challenge.as_slice(), nonce.as_slice()].concat())
            .as_bytes()
            .to_owned();
        #[cfg(not(feature = "solana"))]
        let b = blake3::hash(a.as_slice()).as_bytes().to_owned();

        // Recursive blake3 hash for solana runtime
        #[cfg(feature = "solana")]
        let a = solana_program::blake3::hashv(&[challenge.as_slice(), nonce.as_slice()]).0;
        #[cfg(feature = "solana")]
        let b = solana_program::blake3::hashv(a.as_slice()).0;

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

    /// Build 64-bit digest
    pub fn drill(&mut self) -> [u8; DIGEST_SIZE] {
        // Build digest
        let mut result = [0; DIGEST_SIZE];
        for i in 0..DIGEST_SIZE {
            while !self.exec() {
                self.shuffle(self.opcount as u8 ^ self.state[0]);
                self.opcount += 1;
                if self.opcount.ge(&OPCOUNT_LIMIT) {
                    break;
                }
            }
            result[i] = self.noise[self.addr()];
            self.opcount = 0;
        }

        // Print timers
        println!("Addr {} ns", self.t1);
        println!("Noise {} ns", self.t3);
        println!("Shuffle {} ns", self.t2);
        result
    }

    /// Execute an arithmetic operation
    fn exec(&mut self) -> bool {
        // Do arithmetic
        let noise = self.noise_loop::<64>();
        let mut x = self.state[0] ^ noise[0];
        for i in 0..64 {
            x = self.op(x, self.noise[i]);
            self.state[i] = self.op(self.state[i], x);
        }

        // Shuffle and exit
        self.shuffle(x);
        self.exit()
    }

    /// Derive random address in noise memory space
    pub fn addr(&mut self) -> usize {
        let t = Instant::now();

        // Fill buffer
        let mut addr = [0u8; 8];
        let mut b = 0u8;
        let count = (self.state[3] ^ self.state[1] ^ self.state[41] ^ self.state[59]) as usize;
        for i in 0..count.max(8) {
            addr[i % 8] ^= self.state[b as usize % 64];
            b ^= addr[i % 8];
        }

        // Return
        self.t1 += t.elapsed().as_nanos();
        self.shuffle(b);
        usize::from_le_bytes(addr) % self.noise.len()
    }

    /// Shuffle the internal state
    fn shuffle(&mut self, mut m: u8) {
        let t = Instant::now();

        // Initialize the modifier value
        for &x in &self.state {
            m = m.wrapping_add(x);
        }

        // Unpredictably shuffle every byte of state
        for i in 0..64 {
            // Update state
            let a = self.state[i];
            let index_a = (a as usize ^ m as usize) % 64; // Use xor with the modifier for index
            let b = self.state[index_a];
            let index_b = (b as usize ^ (i * m as usize) as usize) % 64; // Use index and modifier
            self.state[i] = a ^ b ^ self.state[index_b];

            // Introduce some non-linearity
            m = m.wrapping_mul(SHUFFLE_PRIME).wrapping_add(self.state[i]);
        }

        self.t2 += t.elapsed().as_nanos();
    }

    /// Read a slice of noise in a looping unpredictable manner
    fn noise_loop<const N: usize>(&mut self) -> [u8; N] {
        let t = Instant::now();

        // Fill the noise buffer
        let mut result = [0u8; N];
        let mut b = 0u8;
        for i in 0..N {
            let n = self.noise[self.addr()];
            result[i % N] = n ^ self.state[n as usize % 64];
            b = b.wrapping_add(result[i % N]);
        }

        // Shuffle and return
        self.t3 += t.elapsed().as_nanos();
        self.shuffle(b);
        result
    }

    /// Select an opcode from the current state
    fn opcode(&mut self) -> Opcode {
        Opcode::try_from(
            (self.state[2]
                ^ self.state[3]
                ^ self.state[5]
                ^ self.state[7]
                ^ self.state[11]
                ^ self.state[13]
                ^ self.state[17]
                ^ self.state[19])
                % Opcode::cardinality() as u8,
        )
        .expect("Unknown opcode")
    }

    /// Execute a random operation the given operands
    fn op(&mut self, a: u8, b: u8) -> u8 {
        match self.opcode() {
            Opcode::Add => a.wrapping_add(b),
            Opcode::Sub => a.wrapping_sub(b),
            Opcode::Mul => a.wrapping_mul(b),
            Opcode::Div => a.wrapping_div(b.saturating_add(2)),
            Opcode::Nop => a,
            Opcode::Swap => b,
        }
    }

    /// Stop executing on the current byte of the digest
    fn exit(&mut self) -> bool {
        u64::from_be_bytes([
            self.state[23],
            self.state[29],
            self.state[31],
            self.state[37],
            self.state[41],
            self.state[43],
            self.state[47],
            self.state[53],
        ]) % EXIT_OPERAND as u64
            == self.exit.into()
    }
}

/// Set of arbitrary compute operations to chose from
// TODO Sqrt, Floats ?
#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
enum Opcode {
    Add = 0,
    Sub,
    Mul,
    Div,
    Nop,
    Swap,
}

impl Opcode {
    pub fn cardinality() -> usize {
        6
    }
}
