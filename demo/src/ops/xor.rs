use crate::{exit, read_noise};

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
        // Pre-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Xor
        *addr ^= self.b;

        // Post-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Exit
        exit(addr)
    }

    fn update_state(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) {
        self.b ^= u64::from_le_bytes(read_noise(addr, challenge, nonce, noise));
    }
}
