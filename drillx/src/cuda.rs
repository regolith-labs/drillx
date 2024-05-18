extern "C" {
    pub static BATCH_SIZE: u32;
    pub fn hash(challenge: *const u8, nonce: *const u8, out: *mut u64);
}

#[cfg(test)]
mod tests {
    use super::*;

    const INDEX_SPACE: usize = 65536;

    #[test]
    fn test_gpu() {
        let challenge = [255; 32];
        let nonce = [2; 8];
        let mut hashes = vec![0u64; BATCH_SIZE as usize * INDEX_SPACE];
        unsafe {
            hash(
                challenge.as_ptr(),
                nonce.as_ptr(),
                hashes.as_mut_ptr() as *mut u64,
            );
            println!("Got hash: {:?}", hashes[0]);
        }
        assert!(false);
        // let solution = crate::Solution::new(digest, nonce);
        // assert!(solution.is_valid(&challenge));
    }
}
