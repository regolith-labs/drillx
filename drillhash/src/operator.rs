use std::time::Instant;

use num_enum::{IntoPrimitive, TryFromPrimitive};

const OPCOUNT_LIMIT: u32 = 256;
const EXIT_OPERAND: u8 = 17;

#[derive(Debug)]
pub struct Operator<'a> {
    // challenge: &'a [u8; 32],
    // nonce: &'a [u8; 8],
    noise: &'a [u8],
    state: [u8; 64], // 512 bit internal state
    opcount: u32,
    exit: u8,
    t1: u128,
    t2: u128,
    t3: u128,
}

// TODO Sqrt, Floats, Swap ?
#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
enum Opcode {
    Add = 0,
    Sub,
    Mul,
    Div,
    Swap,
}

impl Opcode {
    pub fn cardinality() -> usize {
        5
    }
}

impl<'a> Operator<'a> {
    // Initialize 512 bit internal state from two recursive blake hashes
    pub fn new(challenge: &'a [u8; 32], nonce: &'a [u8; 8], noise: &'a [u8]) -> Operator<'a> {
        #[cfg(feature = "solana")]
        let a = solana_program::blake3::hashv(&[challenge.as_slice(), nonce.as_slice()]).0;
        #[cfg(feature = "solana")]
        let b = solana_program::blake3::hashv(a.as_slice()).0;

        #[cfg(not(feature = "solana"))]
        let a = blake3::hash(&[challenge.as_slice(), nonce.as_slice()].concat())
            .as_bytes()
            .to_owned();
        #[cfg(not(feature = "solana"))]
        let b = blake3::hash(a.as_slice()).as_bytes().to_owned();

        // Build state
        let mut state = [0u8; 64];
        for i in 0..32 {
            state[i * 2] = a[i];
            state[i * 2 + 1] = b[i];
        }

        Operator {
            noise,
            state,
            opcount: 0,
            exit: (state[0] ^ state[1]) % EXIT_OPERAND,
            t1: 0,
            t2: 0,
            t3: 0,
        }
    }

    // Build 64 bit digest
    pub fn drill(&mut self) -> [u8; 64] {
        let mut result = [0; 64];
        for i in 0..64 {
            while !self.exec() {
                self.shuffle(self.opcount as u8 ^ self.state[0]);
                self.opcount += 1;
                if self.opcount.ge(&OPCOUNT_LIMIT) {
                    break;
                }
            }
            self.opcount = 0;
            result[i] = self.noise[self.addr()];
        }

        // println!("Addr time: {} ns", self.t1);
        // println!("Shuffle time: {} ns", self.t2);
        // println!("Noiseloop time: {} ns", self.t3);
        result
    }

    fn exec(&mut self) -> bool {
        // Arithmetic
        // let t = Instant::now();
        let noise = self.noise_loop::<64>();
        let mut b = self.state[0] ^ noise[0];
        for i in 0..64 {
            b = self.op(b, self.noise[i]);
            self.state[i] = self.op(self.state[i], b);
        }

        // Exit code
        // println!("exec in {} nanos", t.elapsed().as_nanos());
        self.shuffle(b);
        self.exit()
    }

    /// Derive address for reading noise file
    pub fn addr(&mut self) -> usize {
        // let t = Instant::now();
        let mut addr = [0u8; 8];
        let count = self.state[63] as usize;

        // Fill addr buffer
        let mut b = 0u8;
        for i in 0..count.max(8) {
            addr[i % 8] ^= self.state[b as usize % 64];
            b ^= addr[i % 8];
        }

        // Return
        // self.t1 += t.elapsed().as_nanos();
        self.shuffle(b);
        usize::from_le_bytes(addr) % self.noise.len()
    }

    fn shuffle(&mut self, mut m: u8) {
        // let t = Instant::now();
        // Accumulate changes in a wrapping manner
        for &value in &self.state {
            m = m.wrapping_add(value);
        }

        for i in 0..64 {
            let a = self.state[i];
            let index_a = (a as usize ^ m as usize) % 64; // Use xor with the modifier for index
            let b = self.state[index_a];
            let index_b = (b as usize ^ (i * m as usize) as usize) % 64; // Use index and modifier

            // More complex operation that depends on previous and next state values
            self.state[i] = a ^ b ^ self.state[index_b];

            // Introduce more non-linearity
            m = m.wrapping_mul(31).wrapping_add(self.state[i]);
        }

        // self.t2 += t.elapsed().as_nanos();
    }

    fn opcode(&mut self) -> Opcode {
        // TODO
        let opcode = self.state[0];
        // let noise = self.noise_loop::<8>();
        // let count = self.state[noise[opcode as usize % 8] as usize % 64];

        // // Derive opcode
        // for _ in 0..count {
        //     // Bitop
        //     opcode ^= self.state[opcode as usize % 64] ^ noise[opcode as usize % 8];

        //     // Shuffle
        //     self.shuffle(opcode);
        // }

        // Return
        Opcode::try_from(opcode % Opcode::cardinality() as u8).expect("Unknown opcode")
    }

    // Loop through noise file
    fn noise_loop<const N: usize>(&mut self) -> [u8; N] {
        // let t = Instant::now();
        let mut result = [0u8; N];

        // Fill the noise buffer
        let mut b = 0u8;
        for i in 0..N {
            let n = self.noise[self.addr()];
            result[i % N] = n ^ self.state[n as usize % 64];
            b = b.wrapping_add(result[i % N]);
        }

        // self.t3 += t.elapsed().as_nanos();
        self.shuffle(b);
        result
    }

    fn op(&mut self, a: u8, b: u8) -> u8 {
        match self.opcode() {
            Opcode::Add => a.wrapping_add(b),
            Opcode::Sub => a.wrapping_sub(b),
            Opcode::Mul => a.wrapping_mul(b),
            Opcode::Div => a.wrapping_div(b.saturating_add(2)),
            Opcode::Swap => b,
        }
    }

    fn exit(&mut self) -> bool {
        let x = u64::from_be_bytes([
            self.state[63],
            self.state[62],
            self.state[61],
            self.state[60],
            self.state[59],
            self.state[58],
            self.state[57],
            self.state[56],
        ]);
        x % EXIT_OPERAND as u64 == self.exit.into()
    }
}
