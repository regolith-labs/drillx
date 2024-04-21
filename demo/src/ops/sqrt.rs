use crate::{exit, read_noise};

use super::Op;

#[derive(Debug)]
pub struct Sqrt {
    mask: u64,
}

impl Default for Sqrt {
    fn default() -> Sqrt {
        Sqrt {
            mask: 0b_10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010,
        }
    }
}

impl Op for Sqrt {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Update mask
        let seed = read_noise(addr, challenge, nonce, noise);
        self.mask ^= u64::from_le_bytes(seed);

        // Apply mask and add
        let addr_ = f64::from_le_bytes((*addr & self.mask).to_le_bytes());
        *addr = u64::from_le_bytes(addr_.sqrt().to_le_bytes());

        // Exit code
        exit(addr)
    }
}
