use crate::{exit, read_value};

use super::Op;

#[derive(Debug)]
pub struct Div {
    b: u64,
}

impl Default for Div {
    fn default() -> Div {
        Div {
            b: u64::from_le_bytes(core::array::from_fn(|i| b"ore"[i % 3])),
        }
    }
}

impl Op for Div {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Update b
        let seed = read_value(addr, challenge, nonce, noise);
        self.b ^= u64::from_le_bytes(seed);

        // Divide
        *addr = addr.wrapping_div(self.b.saturating_add(1));

        // Exit code
        exit(addr)
    }
}
