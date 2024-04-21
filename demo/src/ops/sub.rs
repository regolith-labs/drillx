use crate::{exit, read_noise};

use super::Op;

#[derive(Debug)]
pub struct Sub {
    mask: u64,
    b: u64,
    r: u8,
}

impl Default for Sub {
    fn default() -> Sub {
        Sub {
            mask: 0b_10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010,
            b: u64::from_le_bytes(core::array::from_fn(|i| b"ore"[i % 3])),
            r: 0b_01010101,
        }
    }
}

impl Op for Sub {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Pre-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Subtract
        *addr = (addr.rotate_right(self.r.into()) ^ self.mask).wrapping_sub(self.b);

        // Post-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Exit code
        exit(addr)
    }

    fn update_state(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) {
        self.mask = self.mask.rotate_right(1);
        self.mask ^= u64::from_le_bytes(read_noise(addr, challenge, nonce, noise));
        self.b = self.b.rotate_right(1);
        self.b ^= u64::from_le_bytes(read_noise(addr, challenge, nonce, noise));
        self.r ^= self.mask.to_le_bytes()[7] ^ self.b.to_le_bytes()[7];
    }
}
