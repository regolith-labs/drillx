extern "C" {
    pub fn hash(challenge: *const u8, nonce: *const u8, out: *mut u8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_gpu() {
        let challenge = [255; 32];
        let nonce = [2; 8];
        let mut digest = [0; 16];
        let timer = Instant::now();
        unsafe {
            hash(challenge.as_ptr(), nonce.as_ptr(), digest.as_mut_ptr());
        }
        let solution = crate::Solution::new(digest, nonce);
        println!("Did 1 hash in {} ns", timer.elapsed().as_nanos());
        assert!(!solution.is_valid(&challenge));
    }
}
