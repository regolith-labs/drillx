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

    pub fn hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = VerusHashV2::new();
        hasher.write(data);
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

#[cfg(feature = "solana")]
fn haraka256(input: &mut [u8; 32], output: &mut [u8; 32]) {
    haraka_bpf::haraka256::<5>(output, &input.clone());
}

#[cfg(feature = "solana")]
fn haraka512(input: &mut [u8; 64], output: &mut [u8; 64]) {
    haraka_bpf::haraka512::<5>(output, &input.clone());
}

#[cfg(not(feature = "solana"))]
fn haraka256(input: &mut [u8; 32], output: &mut [u8; 32]) {
    haraka::haraka256::<5>(output, &input.clone());
}

#[cfg(not(feature = "solana"))]
fn haraka512(input: &mut [u8; 64], output: &mut [u8; 64]) {
    haraka::haraka512::<5>(output, &input.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verus_hash() {
        let data = b"Test1234Test1234Test1234Test1234Test1234Test1234Test1234Test1234Test1234Test1234Test1234Test1234";
        let data2 = b"idkidkikdidkidkikdidkidkikdidkidkikdidkidkikdidkidkikdidkidkikdidkidkikdidkidkikdidkidkikd";
        let solution = VerusHashV2::hash(data);
        let solution2 = VerusHashV2::hash(data2);
        let reversed_hex = hex::encode(solution);
        let reversed_hex2 = hex::encode(solution2);
        println!("Solution: {}", reversed_hex);
        println!("Solution2: {}", reversed_hex2);
    }
}
