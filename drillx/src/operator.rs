use std::time::Instant;

use num_enum::{IntoPrimitive, TryFromPrimitive};

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

    /// Counters
    buf: u128,
    op: u128,
    noisec: u128,
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
            buf: 0,
            op: 0,
            noisec: 0,
        }
    }

    /// Build digest using unpredictable and non-parallelizable operations
    pub fn drill(&mut self) -> [u8; DIGEST_SIZE] {
        let mut r = self.state[0];
        let mut result = [0; DIGEST_SIZE];
        for i in 0..DIGEST_SIZE {
            while !self.update() {}
            let addr = usize::from_le_bytes(self.buf(self.opcount as u8));
            let n = self.noise[addr % self.noise.len()];
            r = r.wrapping_add(n);
            result[i] = n.rotate_right(r as u32 % 8);
            self.opcount = 0;
        }

        // Print timers
        println!(
            "Noise {} calls – {} ns – {} ns avg",
            self.t1,
            self.noisec,
            self.t1.saturating_div(self.noisec)
        );
        println!(
            "Op {} calls – {} ns – {} ns avg",
            self.t2,
            self.op,
            self.t2.saturating_div(self.op)
        );
        println!(
            "Buf {} calls – {} ns – {} ns avg",
            self.t3,
            self.buf,
            self.t3.saturating_div(self.buf)
        );
        result
    }

    /// Do unpredictable number of arithmetic operations on internal state
    fn update(&mut self) -> bool {
        // Do arithmetic
        let mut seed = self.state[0];
        for i in 0..64 {
            let n = self.noise::<64>();
            // TODO Loop an unpredictable number of times (combinations of op branches must exceed what brute force can reasonably do)
            // for _ in 0..8 {
            //     //n.max(8) {
            //     n ^= dbg!(self.op(dbg!(n), dbg!(x)));
            //     x ^= dbg!(self.op(dbg!(x), dbg!(n)));
            // }
            let a = self.state[n.wrapping_add(seed) as usize % 64];
            let r = self.state[a.wrapping_add(seed) as usize % 64];
            self.state[i as usize] ^= self.op(a, n).rotate_right(r as u32);
            seed ^= self.state[i as usize];
        }

        // Exit
        self.exit(seed)
    }

    /// Build a buffer of arbitrary size from a seed and internal state
    // TODO Reduce buf calls. Buf itself is pretty cheap (~700 ns), but we call it twice as often as op
    fn buf(&mut self, seed: u8) -> [u8; 8] {
        let t = Instant::now();
        let mut buf = [0u8; 8];
        let mut a = self.state[seed as usize % 64];
        for i in 0..8 {
            buf[i] = a ^ self.state[a as usize % 64];
            a ^= buf[i].rotate_right(a as u32 % 8);
        }
        self.t3 += t.elapsed().as_nanos();
        self.buf += 1;
        buf
    }

    // fn quickbuf(&mut self, r: u8) -> [u8; 8] {
    //     let t = Instant::now();
    //     let mut buf = [0; 8];
    //     let mut a = self.state[r as usize % 64];
    //     for i in 0..8 {
    //         // buf[i] = (self.state[i * 8]
    //         //     ^ self.state[i * 8 + 1]
    //         //     ^ self.state[i * 8 + 2]
    //         //     ^ self.state[i * 8 + 3]
    //         //     ^ self.state[i * 8 + 4]
    //         //     ^ self.state[i * 8 + 5]
    //         //     ^ self.state[i * 8 + 6]
    //         //     ^ self.state[i * 8 + 7])
    //         //     .rotate_right(r as u32)
    //         buf[i] = self.state[a.wrapping_mul(r).wrapping_add(i as u8) as usize % 64];
    //         a ^= buf[i];
    //     }
    //     self.t3 += t.elapsed().as_nanos();
    //     self.buf += 1;
    //     buf
    // }

    /// Read a slice of noise in a looping and unpredictable manner
    fn noise<const N: usize>(&mut self) -> u8 {
        let t = Instant::now();

        // Fill the noise buffer
        // TODO Make this cheaper? Do we need 2 buf calls?
        //      work done in 4675261984 nanos
        //      Noise 534242027 calls 35648 ns 14986 ns avg
        //      Op 23731166 calls 35648 ns 665 ns avg
        //      Buf 53349538 calls 71328 ns 747 ns avg
        //      drill in 604834005 nanos
        //      hash in 110673 nanos
        let mut result = self.sum();
        let mask = usize::from_le_bytes(self.buf(result));
        let mut addr = usize::from_le_bytes(self.buf(mask as u8));
        for _ in 0..N {
            // TODO Test alternative method of addr construction that doesn't rely on hardcoded rotations
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
        self.t1 += t.elapsed().as_nanos();
        self.noisec += 1;
        result
    }

    /// Select an opcode from the current state
    fn opcode(&mut self) -> Opcode {
        Opcode::try_from(self.state[self.opcount as usize % 64] % Opcode::cardinality() as u8)
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
        self.op += 1;
        x
    }

    /// Returns wrapping sum of all 512 bytes in internal state
    fn sum(&self) -> u8 {
        let mut sum = 0u8;
        for x in self.state {
            sum = sum.wrapping_add(x);
        }
        sum
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
