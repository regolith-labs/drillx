use std::time::Instant;

use num_enum::{IntoPrimitive, TryFromPrimitive};

#[derive(Debug)]
pub struct Operator<'a> {
    // challenge: &'a [u8; 32],
    // nonce: &'a [u8; 8],
    noise: &'a [u8],
    state: [u8; 64], // 512 bit internal state
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

        Operator {
            noise,
            state: [
                a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3], a[4], b[4], a[5], b[5], a[6], b[6],
                a[7], b[7], a[8], b[8], a[9], b[9], a[10], b[10], a[11], b[11], a[12], b[12],
                a[13], b[13], a[14], b[14], a[15], b[15], a[16], b[16], a[17], b[17], a[18], b[18],
                a[19], b[19], a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23], a[24], b[24],
                a[25], b[25], a[26], b[26], a[27], b[27], a[28], b[28], a[29], b[29], a[30], b[30],
                a[31], b[31],
            ],
        }
    }

    // Build 64 bit digest
    pub fn drill(&mut self) -> [u8; 64] {
        let mut result = [0; 64];
        for i in 0..64 {
            while !self.exec() {
                self.shuffle(self.state[63]);
            }
            result[i] = self.noise[self.addr()];
        }
        result
    }

    fn exec(&mut self) -> bool {
        // Arithmetic
        // let t = Instant::now();
        self.shuffle(0);
        let noise = self.noise_loop::<64>();
        let mut b = self.state[0] ^ noise[0];
        for i in 0..64 {
            b = self.op(b, self.noise[i]);
            self.state[i] = self.op(self.state[i], b);
            self.shuffle(b);
        }

        // Exit code
        // println!("exec in {} nanos", t.elapsed().as_nanos());
        self.exit()
    }

    /// Derive address for reading noise file
    pub fn addr(&mut self) -> usize {
        // let t = Instant::now();
        let mut addr = [0u8; 8];
        let count = self.state[0] as usize;

        // Fill addr buffer
        let mut b = 0u8;
        for i in 0..count.max(8) {
            // Bitop
            addr[i % 8] ^= self.state[addr[i % 8] as usize % 64];
            b ^= addr[i % 8];
        }

        // Return
        self.shuffle(b);
        // println!("addr in {} nanos", t.elapsed().as_nanos());
        usize::from_le_bytes(addr) % self.noise.len()
    }

    fn shuffle(&mut self, mut m: u8) {
        // let t = Instant::now();
        // First, we'll introduce a random-like deterministic modifier
        for &value in &self.state {
            m = m.wrapping_add(value); // Accumulate changes in a wrapping manner
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
        // println!("shuffle in {} nanos", t.elapsed().as_nanos());
    }

    fn opcode(&mut self) -> Opcode {
        let mut opcode = self.state[0];
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
            // Bitop
            let n = self.noise[self.addr()];
            result[i % N] = n ^ self.state[n as usize % 64];
            b = b.wrapping_add(result[i % N]);
            self.shuffle(b);
        }

        // println!("noiseloop in {} nanos", t.elapsed().as_nanos());
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
        // TODO
        // println!("State: {:?}", self.state);
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
        // println!("X: {:?} {}", x, x % 17);

        x % 17 == 5
        // true
    }
}
