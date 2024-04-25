use fixed::{types::extra::U3, FixedU8};
use num_enum::{IntoPrimitive, TryFromPrimitive};

// TODO Maybe make all consts into variables (could be useful to tune later for onchain performance)

/// Modulo operand for exit condition
const EXIT_OPERAND: u8 = 3;

/// Number of rounds to do before returning the digest
const ROUNDS: usize = 8;

/// How many reads to do per round
const READS_PER_ROUND: usize = 256;

/// How many ops to do per round
const OPS_PER_ROUND: usize = 128;

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
            // TODO
            a = [0; 32]; // blake3::hash(&[challenge.as_slice(), nonce.as_slice()].concat()).as_bytes();
            b = [0; 32]; // blake3::hash(a.as_slice()).as_bytes();
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
    pub fn drill(&mut self) -> [u8; 64] {
        for round in 0..ROUNDS {
            while !self.do_round(round) {}
            self.opcount = 0;
        }
        self.state
    }

    /// Do looping memory reads and arithmetic operations to update internal state
    fn do_round(&mut self, round: usize) -> bool {
        // Do reads
        let mut r = self.state[round % 64];
        for i in 0..READS_PER_ROUND {
            let idx = i * 4 % 64;
            let addr = u32::from_le_bytes(self.state[idx..(idx + 4)].try_into().unwrap())
                .rotate_right(r as u32 % 32);
            self.state[i % 64] ^= self.noise[addr as usize % self.noise.len()];
            r ^= self.state[i % 64];
        }

        // Do ops
        for i in 0..OPS_PER_ROUND {
            r ^= self.op(self.state[i % 64], r);
        }

        // Exit
        self.exit(r)
    }

    /// Select an opcode from the current state
    #[inline]
    fn opcode(&mut self) -> Opcode {
        Opcode::try_from(self.state[self.opcount as usize % 64] % Opcode::cardinality() as u8)
            .expect("Unknown opcode")
    }

    /// Execute a random operation the given operands
    // TODO Avoid case where wrapping_div goes to zero if b > a
    #[inline]
    fn op(&mut self, a: u8, b: u8) -> u8 {
        let x = match self.opcode() {
            Opcode::Add => a.wrapping_add(b),
            Opcode::Sub => a.wrapping_sub(b),
            Opcode::Mul => a.wrapping_mul(b),
            Opcode::Div => {
                if a.gt(&b) {
                    a.wrapping_div(b.saturating_add(2))
                } else {
                    b.wrapping_div(a.saturating_add(2))
                }
            }
            Opcode::Xor => a ^ b,
            Opcode::Right => a >> (b % 8),
            Opcode::Left => a << (b % 8),
            // Opcode::Swap => b,
            // Opcode::Nop => a,
            // Opcode::FAdd => FixedU8::<U3>::from_bits(a)
            //     .wrapping_add(FixedU8::<U3>::from_bits(b))
            //     .to_bits(),
            // Opcode::FSub => FixedU8::<U3>::from_bits(a)
            //     .wrapping_sub(FixedU8::<U3>::from_bits(b))
            //     .to_bits(),
            // Opcode::FMul => FixedU8::<U3>::from_bits(a)
            //     .wrapping_mul(FixedU8::<U3>::from_bits(b))
            //     .to_bits(),
            // Opcode::FSqrt => FixedU8::<U3>::from_bits(a)
            //     .wrapping_sqrt()
            //     .wrapping_add(FixedU8::<U3>::from_bits(b))
            //     .to_bits(),
            // Opcode::Div => a.wrapping_div(b.saturating_add(2)),
        };
        self.opcount += 1;
        x
    }

    /// Stop executing on the current byte of the digest
    fn exit(&mut self, seed: u8) -> bool {
        seed % EXIT_OPERAND == self.exit
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
    Div,
    Xor,
    Right,
    Left,
    // Swap,
    // Nop,
    // FAdd,
    // FSub,
    // FMul,
    // FSqrt,
}

impl Opcode {
    pub fn cardinality() -> usize {
        7
    }
}
