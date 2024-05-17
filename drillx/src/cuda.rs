extern "C" {
    pub fn hash(challenge: *const u8, nonce: *const u8, out: *mut u8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_gpu() {
        let challenge = [1; 32];
        let nonce = [2; 8];
        let mut out = [0; 16];
        let timer = Instant::now();
        unsafe {
            hash(challenge.as_ptr(), nonce.as_ptr(), out.as_mut_ptr());
        }
        println!("Did 65536 hashx in {} ns", timer.elapsed().as_nanos());
        assert_eq!(42, u128::from_le_bytes(out));
    }
}
