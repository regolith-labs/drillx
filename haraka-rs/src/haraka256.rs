use crate::constants;
use crate::simd128::Simd128;
use arrayref::{array_mut_ref, array_ref};

#[inline(always)]
fn aes2(s0: &mut Simd128, s1: &mut Simd128, rci: usize) {
    Simd128::aesenc(s0, &constants::HARAKA_CONSTANTS[rci]);
    Simd128::aesenc(s1, &constants::HARAKA_CONSTANTS[rci + 1]);
    Simd128::aesenc(s0, &constants::HARAKA_CONSTANTS[rci + 2]);
    Simd128::aesenc(s1, &constants::HARAKA_CONSTANTS[rci + 3]);
}

#[inline(always)]
fn mix2(s0: &mut Simd128, s1: &mut Simd128) {
    let mut tmp = *s0;
    Simd128::unpackhi_epi32(&mut tmp, s1);
    Simd128::unpacklo_epi32(s0, s1);
    *s1 = tmp;
}

#[inline(always)]
fn aes_mix2(s0: &mut Simd128, s1: &mut Simd128, rci: usize) {
    aes2(s0, s1, rci);
    mix2(s0, s1);
}

pub fn haraka256<const N_ROUNDS: usize>(dst: &mut [u8; 32], src: &[u8; 32]) {
    let mut s0 = Simd128::read(array_ref![src, 0, 16]);
    let mut s1 = Simd128::read(array_ref![src, 16, 16]);

    for i in 0..N_ROUNDS {
        aes_mix2(&mut s0, &mut s1, 4 * i);
    }

    let t0 = Simd128::read(array_ref![src, 0, 16]);
    let t1 = Simd128::read(array_ref![src, 16, 16]);
    Simd128::pxor(&mut s0, &t0);
    Simd128::pxor(&mut s1, &t1);

    s0.write(array_mut_ref![dst, 0, 16]);
    s1.write(array_mut_ref![dst, 16, 16]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mix2_slice(s0: &mut [u8; 16], s1: &mut [u8; 16]) {
        let mut s0_xmm = Simd128::read(s0);
        let mut s1_xmm = Simd128::read(s1);
        mix2(&mut s0_xmm, &mut s1_xmm);
        s0_xmm.write(s0);
        s1_xmm.write(s1);
    }

    #[test]
    fn test_mix2() {
        let mut dst0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut dst1 = [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let expect0 = [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23];
        let expect1 = [8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31];
        mix2_slice(&mut dst0, &mut dst1);
        assert_eq!(dst0, expect0);
        assert_eq!(dst1, expect1);
    }

    fn aes2_slice(state: &mut [u8; 32], rci: usize) {
        let mut s0_xmm = Simd128::read(array_ref![state, 0, 16]);
        let mut s1_xmm = Simd128::read(array_ref![state, 16, 16]);
        aes2(&mut s0_xmm, &mut s1_xmm, rci);
        s0_xmm.write(array_mut_ref![state, 0, 16]);
        s1_xmm.write(array_mut_ref![state, 16, 16]);
    }

    #[test]
    fn test_aes2() {
        // Test vector computed with https://github.com/kste/haraka/blob/master/code/python/ref.py
        let mut state = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];
        let expect = [
            0xfa, 0xfe, 0x8a, 0x63, 0x12, 0xa6, 0x50, 0x84, 0xf2, 0xbe, 0x62, 0x79, 0x1b, 0x6f,
            0x42, 0x5b, 0x2b, 0x3e, 0xeb, 0x00, 0x60, 0x12, 0x77, 0xab, 0xb1, 0x31, 0x1d, 0x34,
            0x53, 0xd0, 0x90, 0xfc,
        ];
        aes2_slice(&mut state, 0);
        assert_eq!(state, expect);
    }

    fn aes_mix2_slice(state: &mut [u8; 32], rci: usize) {
        let mut s0_xmm = Simd128::read(array_ref![state, 0, 16]);
        let mut s1_xmm = Simd128::read(array_ref![state, 16, 16]);
        aes_mix2(&mut s0_xmm, &mut s1_xmm, rci);
        s0_xmm.write(array_mut_ref![state, 0, 16]);
        s1_xmm.write(array_mut_ref![state, 16, 16]);
    }

    #[test]
    fn test_aes_mix2() {
        // Test vector computed with https://github.com/kste/haraka/blob/master/code/python/ref.py
        let mut state = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];
        let expect = [
            0xfa, 0xfe, 0x8a, 0x63, 0x2b, 0x3e, 0xeb, 0x00, 0x12, 0xa6, 0x50, 0x84, 0x60, 0x12,
            0x77, 0xab, 0xf2, 0xbe, 0x62, 0x79, 0xb1, 0x31, 0x1d, 0x34, 0x1b, 0x6f, 0x42, 0x5b,
            0x53, 0xd0, 0x90, 0xfc,
        ];
        aes_mix2_slice(&mut state, 0);
        assert_eq!(state, expect);
    }

    #[test]
    fn test_haraka256_5round() {
        // Test vector computed with https://github.com/kste/haraka/blob/master/code/python/ref.py
        let mut dst = [0; 32];
        let src = b"\x00\x01\x02\x03\x04\x05\x06\x07\
                    \x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
                    \x10\x11\x12\x13\x14\x15\x16\x17\
                    \x18\x19\x1a\x1b\x1c\x1d\x1e\x1f";
        let expect = b"\x80\x27\xcc\xb8\x79\x49\x77\x4b\
                       \x78\xd0\x54\x5f\xb7\x2b\xf7\x0c\
                       \x69\x5c\x2a\x09\x23\xcb\xd4\x7b\
                       \xba\x11\x59\xef\xbf\x2b\x2c\x1c";
        haraka256::<5>(&mut dst, &src);
        assert_eq!(&dst, expect);
    }

    #[test]
    fn test_haraka256_6round() {
        let mut dst = [0; 32];
        let src = b"\x00\x01\x02\x03\x04\x05\x06\x07\
                    \x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
                    \x10\x11\x12\x13\x14\x15\x16\x17\
                    \x18\x19\x1a\x1b\x1c\x1d\x1e\x1f";
        let expect = b"\xdd\x90\x04\x5b\x92\x99\x32\x74\
                       \xff\xf8\xcc\xf4\x69\x03\xd1\xc8\
                       \x18\x4b\x40\x4c\xc8\x37\x35\x55\
                       \x1c\x80\xa7\x2b\x5f\xb3\x20\x45";
        haraka256::<6>(&mut dst, &src);
        assert_eq!(&dst, expect);
    }

    use std::hint::black_box;
    use test::Bencher;

    fn haraka256_through<const N_ROUNDS: usize>(src: &[u8; 32]) -> [u8; 32] {
        let mut dst = [0; 32];
        haraka256::<N_ROUNDS>(&mut dst, src);
        dst
    }

    #[bench]
    fn bench_haraka256_5round(b: &mut Bencher) {
        let src = b"\x00\x01\x02\x03\x04\x05\x06\x07\
                    \x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
                    \x10\x11\x12\x13\x14\x15\x16\x17\
                    \x18\x19\x1a\x1b\x1c\x1d\x1e\x1f";
        b.iter(|| haraka256_through::<5>(black_box(&src)));
    }

    #[bench]
    fn bench_haraka256_6round(b: &mut Bencher) {
        let src = b"\x00\x01\x02\x03\x04\x05\x06\x07\
                    \x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
                    \x10\x11\x12\x13\x14\x15\x16\x17\
                    \x18\x19\x1a\x1b\x1c\x1d\x1e\x1f";
        b.iter(|| haraka256_through::<6>(black_box(&src)));
    }
}
