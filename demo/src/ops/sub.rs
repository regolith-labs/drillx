use crate::{exit, read_noise};

use super::Op;

#[derive(Debug)]
pub struct Sub {
    b: u64,
}

impl Default for Sub {
    fn default() -> Sub {
        Sub {
            b: u64::from_le_bytes(core::array::from_fn(|i| b"ore"[i % 3])),
        }
    }
}

impl Op for Sub {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Update b
        let seed = read_noise(addr, challenge, nonce, noise);
        self.b ^= u64::from_le_bytes(seed);

        // Subtract
        *addr = addr.wrapping_sub(self.b);

        // Exit code
        exit(addr)
    }
}
