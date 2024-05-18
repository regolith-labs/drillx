extern "C" {
    pub static BATCH_SIZE: u32;
    pub fn hash(challenge: *const u8, nonce: *const u8, out: *mut u64);
    pub fn solve_all_stages(hashes: *const u64, out: *mut u8, sols: *mut u32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

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
            // Do compute heavy hashing on gpu
            let timer = Instant::now();
            hash(
                challenge.as_ptr(),
                nonce.as_ptr(),
                hashes.as_mut_ptr() as *mut u64,
            );
            println!(
                "Gpu returned {} hashes in {} ms",
                BATCH_SIZE,
                timer.elapsed().as_millis()
            );

            // Do memory heavy solution on cpu
            let n = u64::from_le_bytes(nonce);
            for i in 0..BATCH_SIZE as usize {
                let mut digest = [0u8; 16];
                let mut sols = [0u8; 4];
                let batch_start = hashes.as_ptr().add(i * INDEX_SPACE);
                solve_all_stages(
                    batch_start,
                    digest.as_mut_ptr(),
                    sols.as_mut_ptr() as *mut u32,
                );
                if u32::from_le_bytes(sols).gt(&0) {
                    let solution = crate::Solution::new(digest, (n + i as u64).to_le_bytes());
                    assert!(solution.is_valid(&challenge));
                }
            }
            println!(
                "Did {} hashes in {} ms",
                BATCH_SIZE,
                timer.elapsed().as_millis()
            );
            assert!(false);
        }
    }
}
