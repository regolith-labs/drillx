use crate::{exit, read_noise};

use super::Op;

#[derive(Debug)]
pub struct Right {
    b: u64,
}

impl Default for Right {
    fn default() -> Right {
        Right {
            b: u64::from_le_bytes(core::array::from_fn(|i| b"ore"[i % 3])),
        }
    }
}

impl Op for Right {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Pre-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Rotate right
        let b_ = self.b.to_le_bytes();
        let i = b_[0] ^ b_[1] ^ b_[2] ^ b_[3] ^ b_[4] ^ b_[5] ^ b_[6] ^ b_[7];
        *addr = addr.rotate_right(b_[i as usize % 8] as u32);

        // Post-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Exit code
        exit(addr)
    }

    fn update_state(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) {
        self.b ^= u64::from_le_bytes(read_noise(addr, challenge, nonce, noise));
    }
}
