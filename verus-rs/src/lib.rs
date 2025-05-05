pub struct VerusHashV2 {
    buffer: [u8; 128],
    cur_pos: usize,
}

impl VerusHashV2 {
    /// Initializes the VerusHashV2.
    pub fn new() -> Self {
        VerusHashV2 {
            buffer: [0; 128],
            cur_pos: 0,
        }
    }

    pub fn hash(input: &[u8; 40]) -> [u8; 32] {
        let mut hasher = Self::new();
        hasher.write(input);
        hasher.finalize()
    }

    /// Writes data into the hash function, allowing for incremental hashing.
    pub fn write(&mut self, data: &[u8]) {
        let mut pos = 0;
        let len = data.len();

        while pos < len {
            let room = 32 - self.cur_pos;
            let chunk_size = if len - pos >= room { room } else { len - pos };

            self.buffer[32 + self.cur_pos..32 + self.cur_pos + chunk_size]
                .copy_from_slice(&data[pos..pos + chunk_size]);

            self.cur_pos += chunk_size;
            pos += chunk_size;

            if self.cur_pos == 32 {
                let (buf_ptr, buf_ptr2) = self.buffer.split_at_mut(64);
                let buf_ptr: &mut [u8; 64] = buf_ptr.try_into().unwrap();
                let buf_ptr2: &mut [u8; 64] = buf_ptr2.try_into().unwrap();
                haraka512(buf_ptr2, buf_ptr);
                std::mem::swap(buf_ptr, buf_ptr2);
                self.cur_pos = 0;
            }
        }
    }

    /// Finalizes the hashing process and outputs the final hash.
    pub fn finalize(&mut self) -> [u8; 32] {
        if self.cur_pos > 0 {
            self.buffer[32 + self.cur_pos..64].fill(0);
            let (buf_ptr, buf_ptr2) = self.buffer.split_at_mut(64);
            let buf_ptr: &mut [u8; 64] = buf_ptr.try_into().unwrap();
            let buf_ptr2: &mut [u8; 64] = buf_ptr2.try_into().unwrap();
            haraka512(buf_ptr2, buf_ptr);
            std::mem::swap(buf_ptr, buf_ptr2);
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&self.buffer[..32]);
        hash
    }

    /// Resets the internal state of the hash function.
    pub fn reset(&mut self) {
        self.buffer.fill(0);
        self.cur_pos = 0;
    }
}

// // Dummy implementations of Haraka functions for illustration
fn haraka512(input: &mut [u8; 64], output: &mut [u8; 64]) {
    haraka_bpf::haraka512::<6>(output, &input.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verus_hash() {
        let challenge = [0; 40];
        let solution = VerusHashV2::hash(&challenge);
        println!("Solution: {:?}", solution);
    }
}
