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

    pub fn finalize2b(&mut self) -> [u8; 32] {
        panic!("Not implemented");
    }

    pub fn fill_extra(&mut self, data: &[u8]) {
        let mut pos = self.cur_pos;
        let mut remaining = 32 - pos;

        while remaining > 0 {
            let copy_len = remaining.min(data.len());
            self.buffer[32 + pos..32 + pos + copy_len].copy_from_slice(&data[..copy_len]);
            pos += copy_len;
            remaining -= copy_len;
        }
    }
}

// // Dummy implementations of Haraka functions for illustration
fn haraka512(input: &mut [u8; 64], output: &mut [u8; 64]) {
    haraka_bpf::haraka512::<6>(output, &input.clone());
}

pub fn hash(input: &[u8; 40]) -> [u8; 32] {
    let mut hasher = VerusHashV2::new();
    hasher.write(input);
    hasher.finalize()
}

// pub fn hash(result: &mut [u8; 32], data: &[u8]) {
//     let mut buf = [0u8; 128];
//     let (mut buf_ptr, mut buf_ptr2) = buf.split_at_mut(64);
//     let mut pos = 0;
//     let len = data.len();

//     // Initialize the first 32 bytes of the buffer to zero
//     buf_ptr[..32].fill(0);

//     // Process data in chunks of up to 32 bytes
//     while pos < len {
//         if len - pos >= 32 {
//             buf_ptr[32..64].copy_from_slice(&data[pos..pos + 32]);
//         } else {
//             let remaining = len - pos;
//             buf_ptr[32..32 + remaining].copy_from_slice(&data[pos..]);
//             buf_ptr[32 + remaining..64].fill(0);
//         }
//         haraka512(buf_ptr2.try_into().unwrap(), buf_ptr.try_into().unwrap());
//         std::mem::swap(&mut buf_ptr, &mut buf_ptr2);
//         pos += 32;
//     }

//     // Copy the final hash result
//     result.copy_from_slice(&buf_ptr[..32]);
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verus_hash() {
        let data = b"Test1234Test1234Test1234Test1234Test1234Test1234Test1234Test1234Test1234Test1234Test1234Test1234";
        let mut hasher = VerusHashV2::new();
        hasher.reset();
        hasher.write(data.as_slice());
        let solution = hasher.finalize();
        let reversed_hex = hex::encode(solution);
        println!("Solution: {}", reversed_hex);
    }
}
