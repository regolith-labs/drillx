#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    uint8x16_t, vaeseq_u8, vaesmcq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vreinterpretq_u32_u8,
    vreinterpretq_u64_u8, vreinterpretq_u8_u32, vreinterpretq_u8_u64, vst1q_u8, vzip1q_u32,
    vzip1q_u64, vzip2q_u32, vzip2q_u64,
};
#[cfg(target_arch = "arm")]
use core::arch::arm::{
    uint8x16_t, vaeseq_u8, vaesmcq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vreinterpretq_u32_u8,
    vreinterpretq_u8_u32, vst1q_u8, vzipq_u32,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, _mm_aesenc_si128, _mm_loadu_si128, _mm_storeu_si128, _mm_unpackhi_epi32,
    _mm_unpackhi_epi64, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_xor_si128,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, _mm_aesenc_si128, _mm_loadu_si128, _mm_storeu_si128, _mm_unpackhi_epi32,
    _mm_unpackhi_epi64, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_xor_si128,
};
use std::mem::transmute;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub(crate) struct Simd128(__m128i);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Simd128 {
    pub const fn from(x: u128) -> Self {
        Self(unsafe { transmute(x) })
    }

    /// Read from array pointer (potentially unaligned)
    #[inline(always)]
    pub fn read(src: &[u8; 16]) -> Self {
        let x = unsafe { _mm_loadu_si128(src.as_ptr() as *const _ as *const __m128i) };
        Self(x)
    }

    /// Write into array pointer (potentially unaligned)
    #[inline(always)]
    pub fn write(self, dst: &mut [u8; 16]) {
        unsafe {
            _mm_storeu_si128(dst.as_mut_ptr() as *mut _ as *mut __m128i, self.0);
        }
    }

    #[inline(always)]
    pub(crate) fn aesenc(block: &mut Self, key: &Self) {
        unsafe {
            block.0 = _mm_aesenc_si128(block.0, key.0);
        }
    }

    #[inline(always)]
    pub(crate) fn pxor(dst: &mut Self, src: &Self) {
        unsafe {
            dst.0 = _mm_xor_si128(dst.0, src.0);
        }
    }

    #[inline(always)]
    pub(crate) fn unpacklo_epi32(dst: &mut Self, src: &Self) {
        unsafe {
            dst.0 = _mm_unpacklo_epi32(dst.0, src.0);
        }
    }

    #[inline(always)]
    pub(crate) fn unpackhi_epi32(dst: &mut Self, src: &Self) {
        unsafe {
            dst.0 = _mm_unpackhi_epi32(dst.0, src.0);
        }
    }

    #[inline(always)]
    pub(crate) fn unpacklo_epi64(lhs: &Self, rhs: &Self) -> Self {
        unsafe { Self(_mm_unpacklo_epi64(lhs.0, rhs.0)) }
    }

    #[inline(always)]
    pub(crate) fn unpackhi_epi64(lhs: &Self, rhs: &Self) -> Self {
        unsafe { Self(_mm_unpackhi_epi64(lhs.0, rhs.0)) }
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[derive(Clone, Copy)]
pub(crate) struct Simd128(uint8x16_t);

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
impl Simd128 {
    pub const fn from(x: u128) -> Self {
        Self(unsafe { transmute(x) })
    }

    /// Read from array pointer (potentially unaligned)
    #[inline(always)]
    pub fn read(src: &[u8; 16]) -> Self {
        let x = unsafe { vld1q_u8(src.as_ptr() as *const _ as *const u8) };
        Self(x)
    }

    /// Write into array pointer (potentially unaligned)
    #[inline(always)]
    pub fn write(self, dst: &mut [u8; 16]) {
        unsafe {
            vst1q_u8(dst.as_mut_ptr() as *mut _ as *mut u8, self.0);
        }
    }

    #[inline(always)]
    pub(crate) fn aesenc(block: &mut Self, key: &Self) {
        unsafe {
            let zero = vdupq_n_u8(0);
            let x = vaeseq_u8(block.0, zero);
            let y = vaesmcq_u8(x);
            block.0 = veorq_u8(y, key.0);
        }
    }

    #[inline(always)]
    pub(crate) fn pxor(dst: &mut Self, src: &Self) {
        unsafe {
            dst.0 = veorq_u8(dst.0, src.0);
        }
    }

    #[inline(always)]
    pub(crate) fn unpacklo_epi32(dst: &mut Self, src: &Self) {
        unsafe {
            let a = vreinterpretq_u32_u8(dst.0);
            let b = vreinterpretq_u32_u8(src.0);
            // TODO: vzip1q_u32 is missing from core::arch::arm.
            #[cfg(target_arch = "arm")]
            let x = vzipq_u32(a, b).0;
            #[cfg(target_arch = "aarch64")]
            let x = vzip1q_u32(a, b);
            dst.0 = vreinterpretq_u8_u32(x);
        }
    }

    #[inline(always)]
    pub(crate) fn unpackhi_epi32(dst: &mut Self, src: &Self) {
        unsafe {
            let a = vreinterpretq_u32_u8(dst.0);
            let b = vreinterpretq_u32_u8(src.0);
            // TODO: vzip2q_u32 is missing from core::arch::arm.
            #[cfg(target_arch = "arm")]
            let x = vzipq_u32(a, b).1;
            #[cfg(target_arch = "aarch64")]
            let x = vzip2q_u32(a, b);
            dst.0 = vreinterpretq_u8_u32(x);
        }
    }

    // TODO: vzip*q_u64 is missing from core::arch::arm.
    #[cfg(target_arch = "arm")]
    #[inline(always)]
    pub(crate) fn unpacklo_epi64(lhs: &Self, rhs: &Self) -> Self {
        unsafe {
            let a = lhs.split().0;
            let b = rhs.split().0;
            let pair: (u64, u64) = (a, b);
            Self(transmute(pair))
        }
    }

    // TODO: vzip*q_u64 is missing from core::arch::arm.
    #[cfg(target_arch = "arm")]
    #[inline(always)]
    pub(crate) fn unpackhi_epi64(lhs: &Self, rhs: &Self) -> Self {
        unsafe {
            let a = lhs.split().1;
            let b = rhs.split().1;
            let pair: (u64, u64) = (a, b);
            Self(transmute(pair))
        }
    }

    #[cfg(target_arch = "arm")]
    #[inline(always)]
    fn split(&self) -> (u64, u64) {
        unsafe { transmute(self.0) }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub(crate) fn unpacklo_epi64(lhs: &Self, rhs: &Self) -> Self {
        unsafe {
            let a = vreinterpretq_u64_u8(lhs.0);
            let b = vreinterpretq_u64_u8(rhs.0);
            let x = vzip1q_u64(a, b);
            Self(vreinterpretq_u8_u64(x))
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub(crate) fn unpackhi_epi64(lhs: &Self, rhs: &Self) -> Self {
        unsafe {
            let a = vreinterpretq_u64_u8(lhs.0);
            let b = vreinterpretq_u64_u8(rhs.0);
            let x = vzip2q_u64(a, b);
            Self(vreinterpretq_u8_u64(x))
        }
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
        let mut dst = [0u8; 16];
        let key = [0u8; 16];
        let expect = [99u8; 16];
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
