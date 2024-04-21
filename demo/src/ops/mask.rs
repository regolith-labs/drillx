use crate::{exit, read_noise};

use super::Op;

#[derive(Debug)]
pub struct Mask {
    mask: u64,
}

impl Default for Mask {
    fn default() -> Mask {
        Mask {
            mask: 0b_10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010,
        }
    }
}

impl Op for Mask {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Pre-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Mask
        *addr = *addr & self.mask;

        // Post-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Exit code
        exit(addr)
    }

    fn update_state(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) {
        self.mask ^= u64::from_le_bytes(read_noise(addr, challenge, nonce, noise));
    }
}
