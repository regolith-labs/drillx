extern "C" {
    pub static BATCH_SIZE: u32;
    pub fn hash(challenge: *const u8, nonce: *const u8, out: *mut u64);
    pub fn solve_all_stages(hashes: *u64, out: *mut u16);
}

#[cfg(test)]
mod tests {
    use super::*;

    const INDEX_SPACE: usize = 65536;

    fn hashspace_size() -> usize {
        unsafe { BATCH_SIZE as usize * INDEX_SPACE }
    }

    #[test]
    fn test_gpu() {
        let challenge = [255; 32];
        let nonce = [2; 8];
        let mut hashes = vec![0u64; hashspace_size()];
        unsafe {
            hash(
                challenge.as_ptr(),
                nonce.as_ptr(),
                hashes.as_mut_ptr() as *mut u64,
            );
            for i in 0..8 {
                println!("Got hash: {:?}", hashes[i]);
            }
            let mut digest = [0u8; 2];
            unsafe {
                solve_all_stages(hashes.as_ptr(), digest.as_ptr());
                println!("Digest: {:?}", digest);
            }
        }
        assert!(false);
        // let solution = crate::Solution::new(digest, nonce);
        // assert!(solution.is_valid(&challenge));
    }
}
