extern "C" {
    pub fn drillx(challenge: *const u8, nonce: *const u8, out: *mut u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu() {
        let challenge = [1; 32];
        let nonce = [2; 8];
        let mut out = [0; 16];
        unsafe {
            drillx(
                challenge.as_ptr(),
                u64::from_le_bytes(nonce),
                out.as_mut_ptr(),
            );
        }
        assert_eq!(42, u16::from_le_bytes(out));
    }
}
