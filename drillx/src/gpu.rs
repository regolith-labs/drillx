extern "C" {
    pub fn drill_hash(challenge: *const u8, out: *mut u8, secs: u64);
    pub fn single_drill_hash(challenge: *const u8, nonce: u64, out: *mut u8);
    pub fn set_noise(noise: *const usize);
    pub fn get_noise(noise: *const usize);
    pub fn gpu_init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::NOISE;
    use crate::operator2::Operator2;

    #[test]
    fn test_gpu() {
        let mut noise = vec![0_usize; 1000 * 1000 / 8];
        unsafe {
            gpu_init();
            set_noise(NOISE.as_usize_slice().as_ptr());
            get_noise(noise.as_mut_ptr());
        }
        assert_eq!(&noise, NOISE.as_usize_slice());

        let challenge = [1; 32];
        let nonce = [2; 8];
        let mut out = [0; 32];
        unsafe {
            single_drill_hash(
                challenge.as_ptr(),
                u64::from_le_bytes(nonce),
                out.as_mut_ptr(),
            );
        }

        // cpu
        let digest = Operator2::new(&challenge, &nonce).drill();
        let cpu_out = solana_program::keccak::hashv(&[digest.as_slice()]).0;

        assert_eq!(out, cpu_out)
    }
}
