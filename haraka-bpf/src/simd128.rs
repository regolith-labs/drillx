use aes::cipher::Block;
use aes::hazmat::cipher_round;
use aes::Aes128; // Import the specific AES type
use core::ops::BitXorAssign;

/// Represents a 128-bit SIMD value, implemented using aes::Block<aes::Aes128> for portability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct Simd128(Block<Aes128>);

impl Simd128 {
    /// Creates a Simd128 value from a u128.
    /// Note: The byte order depends on the target architecture (little-endian assumed).
    pub const fn from(x: u128) -> Self {
        // GenericArray requires a const context for `from_slice`, which isn't stable yet.
        // We use `to_le_bytes` and transmute as a workaround.
        // This assumes the underlying Block representation matches [u8; 16].
        // SAFETY: Block is repr(transparent) over GenericArray<u8, U16>, which has the same
        // size and alignment as [u8; 16].
        unsafe { core::mem::transmute(x.to_le_bytes()) }
    }

    /// Read from array pointer (potentially unaligned)
    #[inline(always)]
    pub fn read(src: &[u8; 16]) -> Self {
        Self(Block::<Aes128>::clone_from_slice(src))
    }

    /// Write into array pointer (potentially unaligned)
    #[inline(always)]
    pub fn write(self, dst: &mut [u8; 16]) {
        dst.copy_from_slice(self.0.as_slice());
    }

    /// Performs one round of AES encryption (SubBytes, ShiftRows, MixColumns)
    /// on the block, then XORs the result with the key.
    /// This mimics the behavior of the `_mm_aesenc_si128` intrinsic.
    #[inline(always)]
    pub(crate) fn aesenc(block: &mut Self, key: &Self) {
        // cipher_round performs SubBytes, ShiftRows, MixColumns, and AddRoundKey (XOR)
        cipher_round(&mut block.0, &key.0);
    }

    /// Performs a bitwise XOR operation.
    #[inline(always)]
    pub(crate) fn pxor(dst: &mut Self, src: &Self) {
        *dst ^= *src;
    }

    /// Interleaves the lower 4-byte words of `dst` and `src`.
    /// dst = [a0 a1 a2 a3 | a4 a5 a6 a7 | a8 a9 aa ab | ac ad ae af]
    /// src = [b0 b1 b2 b3 | b4 b5 b6 b7 | b8 b9 ba bb | bc bd be bf]
    /// result = [a0 a1 a2 a3 | b0 b1 b2 b3 | a4 a5 a6 a7 | b4 b5 b6 b7]
    #[inline(always)]
    pub(crate) fn unpacklo_epi32(dst: &mut Self, src: &Self) {
        let mut res = [0u8; 16];
        let a = dst.0.as_slice();
        let b = src.0.as_slice();

        res[0..4].copy_from_slice(&a[0..4]);
        res[4..8].copy_from_slice(&b[0..4]);
        res[8..12].copy_from_slice(&a[4..8]);
        res[12..16].copy_from_slice(&b[4..8]);

        dst.0 = Block::<Aes128>::clone_from_slice(&res);
    }

    /// Interleaves the higher 4-byte words of `dst` and `src`.
    /// dst = [a0 a1 a2 a3 | a4 a5 a6 a7 | a8 a9 aa ab | ac ad ae af]
    /// src = [b0 b1 b2 b3 | b4 b5 b6 b7 | b8 b9 ba bb | bc bd be bf]
    /// result = [a8 a9 aa ab | b8 b9 ba bb | ac ad ae af | bc bd be bf]
    #[inline(always)]
    pub(crate) fn unpackhi_epi32(dst: &mut Self, src: &Self) {
        let mut res = [0u8; 16];
        let a = dst.0.as_slice();
        let b = src.0.as_slice();

        res[0..4].copy_from_slice(&a[8..12]);
        res[4..8].copy_from_slice(&b[8..12]);
        res[8..12].copy_from_slice(&a[12..16]);
        res[12..16].copy_from_slice(&b[12..16]);

        dst.0 = Block::<Aes128>::clone_from_slice(&res);
    }

    /// Interleaves the lower 8-byte words of `lhs` and `rhs`.
    /// lhs = [a0..a7 | a8..af]
    /// rhs = [b0..b7 | b8..bf]
    /// result = [a0..a7 | b0..b7]
    #[inline(always)]
    pub(crate) fn unpacklo_epi64(lhs: &Self, rhs: &Self) -> Self {
        let mut res = [0u8; 16];
        let a = lhs.0.as_slice();
        let b = rhs.0.as_slice();

        res[0..8].copy_from_slice(&a[0..8]);
        res[8..16].copy_from_slice(&b[0..8]);

        Self(Block::<Aes128>::clone_from_slice(&res))
    }

    /// Interleaves the higher 8-byte words of `lhs` and `rhs`.
    /// lhs = [a0..a7 | a8..af]
    /// rhs = [b0..b7 | b8..bf]
    /// result = [a8..af | b8..bf]
    #[inline(always)]
    pub(crate) fn unpackhi_epi64(lhs: &Self, rhs: &Self) -> Self {
        let mut res = [0u8; 16];
        let a = lhs.0.as_slice();
        let b = rhs.0.as_slice();

        res[0..8].copy_from_slice(&a[8..16]);
        res[8..16].copy_from_slice(&b[8..16]);

        Self(Block::<Aes128>::clone_from_slice(&res))
    }
}

impl BitXorAssign for Simd128 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        // Perform XOR element-wise since GenericArray doesn't implement BitXorAssign
        for (a, b) in self.0.as_mut_slice().iter_mut().zip(rhs.0.as_slice()) {
            *a ^= *b;
        }
    }
}

// Implement Default for Simd128 if needed, e.g., for constants
impl Default for Simd128 {
    fn default() -> Self {
        Self(Block::<Aes128>::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aesenc_slice(block: &mut [u8; 16], key: &[u8; 16]) {
        let mut block_xmm = Simd128::read(block);
        let key_xmm = Simd128::read(key);
        Simd128::aesenc(&mut block_xmm, &key_xmm);
        block_xmm.write(block);
    }

    #[test]
    fn test_aesenc() {
        // Applying one AES round (SubBytes, ShiftRows, MixColumns) + AddRoundKey (XOR)
        // to a zero block with a zero key results in [0x63; 16].
        // SubBytes(0) = 0x63. ShiftRows and MixColumns on a uniform block yield the same block.
        // XORing with 0 leaves it unchanged.
        let mut dst = [0u8; 16];
        let key = [0u8; 16];
        let expect = [0x63u8; 16]; // Corrected expected value
        aesenc_slice(&mut dst, &key);
        assert_eq!(dst, expect);
    }

    fn pxor_slice(dst: &mut [u8; 16], src: &[u8; 16]) {
        let mut dst_xmm = Simd128::read(dst);
        let src_xmm = Simd128::read(src);
        Simd128::pxor(&mut dst_xmm, &src_xmm);
        dst_xmm.write(dst);
    }

    #[test]
    fn test_pxor() {
        let mut dst = [0xb2u8; 16];
        let src = [0xc5u8; 16];
        let expect = [(0xb2u8 ^ 0xc5u8); 16];
        pxor_slice(&mut dst, &src);
        assert_eq!(dst, expect);
    }

    fn unpacklo_epi32_slice(dst: &mut [u8; 16], src: &[u8; 16]) {
        let mut dst_xmm = Simd128::read(dst);
        let src_xmm = Simd128::read(src);
        Simd128::unpacklo_epi32(&mut dst_xmm, &src_xmm);
        dst_xmm.write(dst);
    }

    #[test]
    fn test_unpacklo_epi32() {
        let mut dst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let src = [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let expect = [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23];
        unpacklo_epi32_slice(&mut dst, &src);
        assert_eq!(dst, expect);
    }

    fn unpackhi_epi32_slice(dst: &mut [u8; 16], src: &[u8; 16]) {
        let mut dst_xmm = Simd128::read(dst);
        let src_xmm = Simd128::read(src);
        Simd128::unpackhi_epi32(&mut dst_xmm, &src_xmm);
        dst_xmm.write(dst);
    }

    #[test]
    fn test_unpackhi_epi32() {
        let mut dst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let src = [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let expect = [8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31];
        unpackhi_epi32_slice(&mut dst, &src);
        assert_eq!(dst, expect);
    }

    fn unpacklo_epi64_slice(lhs: &[u8; 16], rhs: &[u8; 16]) -> [u8; 16] {
        let lhs_xmm = Simd128::read(lhs);
        let rhs_xmm = Simd128::read(rhs);
        let result = Simd128::unpacklo_epi64(&lhs_xmm, &rhs_xmm);
        let mut dst = [0; 16];
        result.write(&mut dst);
        dst
    }

    #[test]
    fn test_unpacklo_epi64() {
        let lhs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let rhs = [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let expect = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
        let unpacked = unpacklo_epi64_slice(&lhs, &rhs);
        assert_eq!(unpacked, expect);
    }

    fn unpackhi_epi64_slice(lhs: &[u8; 16], rhs: &[u8; 16]) -> [u8; 16] {
        let lhs_xmm = Simd128::read(lhs);
        let rhs_xmm = Simd128::read(rhs);
        let result = Simd128::unpackhi_epi64(&lhs_xmm, &rhs_xmm);
        let mut dst = [0; 16];
        result.write(&mut dst);
        dst
    }

    #[test]
    fn test_unpackhi_epi64() {
        let lhs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let rhs = [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let expect = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31];
        let unpacked = unpackhi_epi64_slice(&lhs, &rhs);
        assert_eq!(unpacked, expect);
    }
}
