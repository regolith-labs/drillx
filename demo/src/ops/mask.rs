use crate::{exit, read_value};

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
        // Update mask
        let seed = read_value(addr, challenge, nonce, noise);
        self.mask ^= u64::from_le_bytes(seed);

        // Apply mask and add
        *addr = *addr & self.mask;

        // Exit code
        exit(addr)
    }
}
