use crate::{exit, read_noise};

use super::Op;

#[derive(Debug)]
pub struct MulL {
    mask: u64,
    b: u64,
    l: u8,
}

impl Default for MulL {
    fn default() -> MulL {
        MulL {
            mask: 0b_10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010,
            b: u64::from_le_bytes(core::array::from_fn(|i| b"ore"[i % 3])),
            l: 0b_01010101,
        }
    }
}

impl Op for MulL {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Pre-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Multiply
        *addr = addr
            .saturating_add(2)
            .wrapping_mul(self.b)
            .rotate_left(self.l.into())
            ^ self.mask;
        // *addr = (addr.rotate_left(self.l.into()) ^ self.mask).wrapping_mul(self.b);

        // Post-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Exit code
        exit(addr)
    }

    fn update_state(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) {
        self.mask = self.mask.rotate_left(1);
        self.mask ^= u64::from_le_bytes(read_noise(addr, challenge, nonce, noise));
        self.b = self.b.rotate_left(1);
        self.b ^= u64::from_le_bytes(read_noise(addr, challenge, nonce, noise));
        self.l ^= self.mask.to_le_bytes()[7] ^ self.b.to_le_bytes()[7];
    }
}
