use crate::{exit, read_value};

use super::Op;

#[derive(Debug)]
pub struct Add {
    b: u64,
}

impl Default for Add {
    fn default() -> Add {
        Add {
            b: u64::from_le_bytes(core::array::from_fn(|i| b"ore"[i % 3])),
        }
    }
}

impl Op for Add {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Update b
        let seed = read_value(addr, challenge, nonce, noise);
        self.b ^= u64::from_le_bytes(seed);

        // Add
        *addr = addr.wrapping_add(self.b);

        // Exit code
        exit(addr)
    }
}
