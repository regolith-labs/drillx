extern "C" {
    pub fn hash(challenge: *const u8, nonce: *const u8, out: *mut u64);
}

#[cfg(test)]
mod tests {
    use super::*;

    const BATCH_SIZE: usize = 8;
    const INDEX_SPACE: usize = 65536;
    const HASH_SPACE: usize = BATCH_SIZE * INDEX_SPACE;

    #[test]
    fn test_gpu() {
        let challenge = [255; 32];
        let nonce = [2; 8];
        let mut hashes = [0u64; HASH_SPACE];
        unsafe {
            hash(challenge.as_ptr(), nonce.as_ptr(), hashes.as_mut_ptr());
            println!("Got hash: {:?}", hashes[0]);
        }
        assert!(false);
        // let solution = crate::Solution::new(digest, nonce);
        // assert!(solution.is_valid(&challenge));
    }
}
