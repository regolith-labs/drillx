use crate::{exit, read_value};

use super::Op;

#[derive(Debug)]
pub struct Xor {
    b: u64,
}

impl Default for Xor {
    fn default() -> Xor {
        Xor {
            b: 0b_01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101,
        }
    }
}

impl Op for Xor {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Update b
        let seed = read_value(addr, challenge, nonce, noise);
        self.b ^= u64::from_le_bytes(seed);

        // Xor
        *addr ^= self.b;

        // Exit
        exit(addr)
    }
}
