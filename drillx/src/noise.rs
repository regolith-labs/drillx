#[repr(align(8))]
pub struct Noise<const N: usize>([u8; N]);

pub static NOISE: Noise<1_000_000> = Noise(*include_bytes!("noise.txt"));

impl<const N: usize> Noise<N> {
    // Check if the slice is properly aligned and sized
    pub fn as_usize_slice(&self) -> Option<&[usize]> {
        let align = std::mem::align_of::<usize>();
        let size = std::mem::size_of::<usize>();
        if self.0.as_ptr() as usize % align == 0 && N % size == 0 {
            Some(unsafe { as_usize_slice_unchecked(self.0.as_slice()) })
        } else {
            None
        }
    }
}

unsafe fn as_usize_slice_unchecked(bytes: &[u8]) -> &[usize] {
    let len = bytes.len() / std::mem::size_of::<usize>();
    std::slice::from_raw_parts(bytes.as_ptr() as *const usize, len)
}
