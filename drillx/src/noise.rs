const N: usize = 1_000_000;

#[repr(C, align(8))]
pub struct Noise([u8; N]);

pub static NOISE: Noise = Noise(*include_bytes!("noise.txt"));

const _: () = assert!(core::mem::align_of::<Noise>() == core::mem::align_of::<usize>());
const _: () = assert!(core::mem::size_of::<Noise>() % core::mem::size_of::<usize>() == 0);

impl Noise {
    // Check if the slice is properly aligned and sized
    pub const fn as_usize_slice(&self) -> &[usize] {
        unsafe { as_usize_slice_unchecked(self.0.as_slice()) }
    }
}

const unsafe fn as_usize_slice_unchecked(bytes: &[u8]) -> &[usize] {
    let len = bytes.len() / std::mem::size_of::<usize>();
    std::slice::from_raw_parts(bytes.as_ptr() as *const usize, len)
}
