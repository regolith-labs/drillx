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

            // Do memory heavy work on cpu
            let chunk_size = BATCH_SIZE as usize / num_threads;
            let challenge = Arc::new(challenge);
            let mut handles = vec![];
            for t in 0..num_cpus::get() {
                let hashes = Arc::clone(&hashes);
                let challenge = Arc::clone(&challenge);
                let nonce = u64::from_le_bytes(nonce);
                let handle = thread::spawn(move || {
                    let start = t * chunk_size;
                    let end = if t == num_threads - 1 {
                        BATCH_SIZE as usize
                    } else {
                        start + chunk_size
                    };
                    for i in start..end {
                        let mut digest = [0u8; 16];
                        let mut sols = [0u8; 4];
                        let batch_start = hashes.as_ptr().add(i * INDEX_SPACE);
                        unsafe {
                            solve_all_stages(
                                batch_start,
                                digest.as_mut_ptr(),
                                sols.as_mut_ptr() as *mut u32,
                            );
                        }
                        if u32::from_le_bytes(sols).gt(&0) {
                            let solution =
                                crate::Solution::new(digest, (nonce + i as u64).to_le_bytes());
                            assert!(solution.is_valid(&challenge));
                        }
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().expect("Failed to join thread");
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
