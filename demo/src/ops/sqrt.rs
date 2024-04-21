use crate::{exit, read_noise};

use super::Op;

#[derive(Debug)]
pub struct Sqrt {
    mask: u64,
    r: u8,
}

impl Default for Sqrt {
    fn default() -> Sqrt {
        Sqrt {
            mask: 0b_10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010,
            r: 0b_01010101,
        }
    }
}

impl Op for Sqrt {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool {
        // Pre-arithmetic
        println!("ADDR1: {}", *addr);
        self.update_state(addr, challenge, nonce, noise);
        println!("ADDR2: {}", *addr);

        // Square root
        // println!(
        //     "SQRT {} {} {}",
        //     *addr,
        //     addr.rotate_right(self.r.into()),
        //     addr.rotate_right(self.r.into()) & self.mask,
        // );
        let addr_ = f64::from_le_bytes(addr.to_le_bytes());
        *addr =
            u64::from_le_bytes(addr_.sqrt().to_le_bytes()).rotate_right(self.r.into()) ^ self.mask;

        // Post-arithmetic
        self.update_state(addr, challenge, nonce, noise);

        // Exit code
        exit(addr)
    }

    fn update_state(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) {
        self.mask = self.mask.rotate_right(1);
        self.mask ^= u64::from_le_bytes(read_noise(addr, challenge, nonce, noise));
        self.r ^= self.r.rotate_right(1) ^ self.mask.to_le_bytes()[7];
    }
}
