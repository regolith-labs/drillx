use crate::{exit, read_noise};

use super::Op;

#[derive(Debug)]
pub struct Mul {
    b: u64,
}

impl Default for Mul {
    fn default() -> Mul {
        Mul {
            b: u64::from_le_bytes(core::array::from_fn(|i| b"ore"[i % 3])),
        }
    }
}

impl Op for Mul {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Update b
        let seed = read_noise(addr, challenge, nonce, noise);
        self.b ^= u64::from_le_bytes(seed);

        // Multiply
        *addr = addr.wrapping_mul(self.b);

        // Exit code
        exit(addr)
    }
}
