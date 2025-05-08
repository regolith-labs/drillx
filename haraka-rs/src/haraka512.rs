use crate::constants;
use crate::simd128::Simd128;
use arrayref::{array_mut_ref, array_ref};

#[inline(always)]
fn aes4(s0: &mut Simd128, s1: &mut Simd128, s2: &mut Simd128, s3: &mut Simd128, rci: usize) {
    Simd128::aesenc(s0, &constants::HARAKA_CONSTANTS[rci]);
    Simd128::aesenc(s1, &constants::HARAKA_CONSTANTS[rci + 1]);
    Simd128::aesenc(s2, &constants::HARAKA_CONSTANTS[rci + 2]);
    Simd128::aesenc(s3, &constants::HARAKA_CONSTANTS[rci + 3]);
    Simd128::aesenc(s0, &constants::HARAKA_CONSTANTS[rci + 4]);
    Simd128::aesenc(s1, &constants::HARAKA_CONSTANTS[rci + 5]);
    Simd128::aesenc(s2, &constants::HARAKA_CONSTANTS[rci + 6]);
    Simd128::aesenc(s3, &constants::HARAKA_CONSTANTS[rci + 7]);
}

#[inline(always)]
fn mix4(s0: &mut Simd128, s1: &mut Simd128, s2: &mut Simd128, s3: &mut Simd128) {
    let mut tmp = *s0;
    Simd128::unpacklo_epi32(&mut tmp, s1);
    Simd128::unpackhi_epi32(s0, s1);
    *s1 = *s2;
    Simd128::unpacklo_epi32(s1, s3);
    Simd128::unpackhi_epi32(s2, s3);

    *s3 = *s0;
    Simd128::unpacklo_epi32(s3, s2);
    Simd128::unpackhi_epi32(s0, s2);
    *s2 = *s1;
    Simd128::unpackhi_epi32(s2, &tmp);
    Simd128::unpacklo_epi32(s1, &tmp);
}

#[inline(always)]
fn aes_mix4(s0: &mut Simd128, s1: &mut Simd128, s2: &mut Simd128, s3: &mut Simd128, rci: usize) {
    aes4(s0, s1, s2, s3, rci);
    mix4(s0, s1, s2, s3);
}

#[inline(always)]
fn truncstore(dst: &mut [u8; 32], s0: &Simd128, s1: &Simd128, s2: &Simd128, s3: &Simd128) {
    Simd128::unpackhi_epi64(s0, s1).write(array_mut_ref![dst, 0, 16]);
    Simd128::unpacklo_epi64(s2, s3).write(array_mut_ref![dst, 16, 16]);
}

pub fn haraka512<const N_ROUNDS: usize>(dst: &mut [u8; 32], src: &[u8; 64]) {
    let mut s0 = Simd128::read(array_ref![src, 0, 16]);
    let mut s1 = Simd128::read(array_ref![src, 16, 16]);
    let mut s2 = Simd128::read(array_ref![src, 32, 16]);
    let mut s3 = Simd128::read(array_ref![src, 48, 16]);

    for i in 0..N_ROUNDS {
        aes_mix4(&mut s0, &mut s1, &mut s2, &mut s3, 8 * i);
    }

    let t0 = Simd128::read(array_ref![src, 0, 16]);
    let t1 = Simd128::read(array_ref![src, 16, 16]);
    let t2 = Simd128::read(array_ref![src, 32, 16]);
    let t3 = Simd128::read(array_ref![src, 48, 16]);
    Simd128::pxor(&mut s0, &t0);
    Simd128::pxor(&mut s1, &t1);
    Simd128::pxor(&mut s2, &t2);
    Simd128::pxor(&mut s3, &t3);

    truncstore(dst, &s0, &s1, &s2, &s3);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mix4_slice(s0: &mut [u8; 16], s1: &mut [u8; 16], s2: &mut [u8; 16], s3: &mut [u8; 16]) {
        let mut s0_xmm = Simd128::read(s0);
        let mut s1_xmm = Simd128::read(s1);
        let mut s2_xmm = Simd128::read(s2);
        let mut s3_xmm = Simd128::read(s3);
        mix4(&mut s0_xmm, &mut s1_xmm, &mut s2_xmm, &mut s3_xmm);
        s0_xmm.write(s0);
        s1_xmm.write(s1);
        s2_xmm.write(s2);
        s3_xmm.write(s3);
    }

    #[test]
    fn test_mix4() {
        let mut dst0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut dst1 = [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let mut dst2 = [
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        ];
        let mut dst3 = [
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ];
        let expect0 = [
            12, 13, 14, 15, 44, 45, 46, 47, 28, 29, 30, 31, 60, 61, 62, 63,
        ];
        let expect1 = [32, 33, 34, 35, 0, 1, 2, 3, 48, 49, 50, 51, 16, 17, 18, 19];
        let expect2 = [36, 37, 38, 39, 4, 5, 6, 7, 52, 53, 54, 55, 20, 21, 22, 23];
        let expect3 = [8, 9, 10, 11, 40, 41, 42, 43, 24, 25, 26, 27, 56, 57, 58, 59];
        mix4_slice(&mut dst0, &mut dst1, &mut dst2, &mut dst3);
        assert_eq!(dst0, expect0);
        assert_eq!(dst1, expect1);
        assert_eq!(dst2, expect2);
        assert_eq!(dst3, expect3);
    }

    fn aes4_slice(state: &mut [u8; 64], rci: usize) {
        let mut s0_xmm = Simd128::read(array_ref![state, 0, 16]);
        let mut s1_xmm = Simd128::read(array_ref![state, 16, 16]);
        let mut s2_xmm = Simd128::read(array_ref![state, 32, 16]);
        let mut s3_xmm = Simd128::read(array_ref![state, 48, 16]);
        aes4(&mut s0_xmm, &mut s1_xmm, &mut s2_xmm, &mut s3_xmm, rci);
        s0_xmm.write(array_mut_ref![state, 0, 16]);
        s1_xmm.write(array_mut_ref![state, 16, 16]);
        s2_xmm.write(array_mut_ref![state, 32, 16]);
        s3_xmm.write(array_mut_ref![state, 48, 16]);
    }

    #[test]
    fn test_aes4() {
        // Test vector computed with https://github.com/kste/haraka/blob/master/code/python/ref.py
        let mut state = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
            0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
            0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
        ];
        let expect: [u8; 64] = [
            0xaa, 0x1f, 0xb1, 0x42, 0x6e, 0xf6, 0xbc, 0x32, 0xe1, 0x7e, 0xe2, 0x62, 0xfd, 0x01,
            0x8f, 0xa4, 0x29, 0x74, 0x3a, 0xd6, 0x22, 0x9b, 0xd7, 0x9d, 0x0e, 0x4c, 0xf6, 0x74,
            0x5b, 0xf7, 0xac, 0x8c, 0x1e, 0xa9, 0x8c, 0x29, 0x1f, 0x90, 0x89, 0x90, 0xe9, 0x28,
            0x6a, 0x2b, 0xac, 0x14, 0xe1, 0xad, 0x3a, 0xcd, 0xdf, 0xb2, 0xfa, 0xbe, 0xa4, 0x55,
            0x36, 0x97, 0x60, 0x28, 0xe8, 0x11, 0xbb, 0xfd,
        ];
        aes4_slice(&mut state, 0);
        // TODO: cannot use assert_eq for [u8; 64]
        assert_eq!(&state as &[u8], &expect as &[u8]);
    }

    fn aes_mix4_slice(state: &mut [u8; 64], rci: usize) {
        let mut s0_xmm = Simd128::read(array_ref![state, 0, 16]);
        let mut s1_xmm = Simd128::read(array_ref![state, 16, 16]);
        let mut s2_xmm = Simd128::read(array_ref![state, 32, 16]);
        let mut s3_xmm = Simd128::read(array_ref![state, 48, 16]);
        aes_mix4(&mut s0_xmm, &mut s1_xmm, &mut s2_xmm, &mut s3_xmm, rci);
        s0_xmm.write(array_mut_ref![state, 0, 16]);
        s1_xmm.write(array_mut_ref![state, 16, 16]);
        s2_xmm.write(array_mut_ref![state, 32, 16]);
        s3_xmm.write(array_mut_ref![state, 48, 16]);
    }

    #[test]
    fn test_aes_mix4() {
        // Test vector computed with https://github.com/kste/haraka/blob/master/code/python/ref.py
        let mut state = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
            0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
            0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
        ];
        let expect: [u8; 64] = [
            0xfd, 0x01, 0x8f, 0xa4, 0xac, 0x14, 0xe1, 0xad, 0x5b, 0xf7, 0xac, 0x8c, 0xe8, 0x11,
            0xbb, 0xfd, 0x1e, 0xa9, 0x8c, 0x29, 0xaa, 0x1f, 0xb1, 0x42, 0x3a, 0xcd, 0xdf, 0xb2,
            0x29, 0x74, 0x3a, 0xd6, 0x1f, 0x90, 0x89, 0x90, 0x6e, 0xf6, 0xbc, 0x32, 0xfa, 0xbe,
            0xa4, 0x55, 0x22, 0x9b, 0xd7, 0x9d, 0xe1, 0x7e, 0xe2, 0x62, 0xe9, 0x28, 0x6a, 0x2b,
            0x0e, 0x4c, 0xf6, 0x74, 0x36, 0x97, 0x60, 0x28,
        ];
        aes_mix4_slice(&mut state, 0);
        // TODO: cannot use assert_eq for [u8; 64]
        assert_eq!(&state as &[u8], &expect as &[u8]);
    }

    fn truncstore_slice(dst: &mut [u8; 32], state: &[u8; 64]) {
        let s0_xmm = Simd128::read(array_ref![state, 0, 16]);
        let s1_xmm = Simd128::read(array_ref![state, 16, 16]);
        let s2_xmm = Simd128::read(array_ref![state, 32, 16]);
        let s3_xmm = Simd128::read(array_ref![state, 48, 16]);
        truncstore(dst, &s0_xmm, &s1_xmm, &s2_xmm, &s3_xmm);
    }

    #[test]
    fn test_truncstore() {
        // Test vector computed with https://github.com/kste/haraka/blob/master/code/python/ref.py
        let mut dst = [0u8; 32];
        let state = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
            0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
            0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
        ];
        let expect: [u8; 32] = [
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x30, 0x31, 0x32, 0x33,
            0x34, 0x35, 0x36, 0x37,
        ];
        truncstore_slice(&mut dst, &state);
        assert_eq!(dst, expect);
    }

    #[test]
    fn test_haraka512_5round() {
        // Test vector computed with https://github.com/kste/haraka/blob/master/code/python/ref.py
        let mut dst = [0; 32];
        let src = b"\x00\x01\x02\x03\x04\x05\x06\x07\
                    \x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
                    \x10\x11\x12\x13\x14\x15\x16\x17\
                    \x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\
                    \x20\x21\x22\x23\x24\x25\x26\x27\
                    \x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\
                    \x30\x31\x32\x33\x34\x35\x36\x37\
                    \x38\x39\x3a\x3b\x3c\x3d\x3e\x3f";
        let expect = b"\xbe\x7f\x72\x3b\x4e\x80\xa9\x98\
                       \x13\xb2\x92\x28\x7f\x30\x6f\x62\
                       \x5a\x6d\x57\x33\x1c\xae\x5f\x34\
                       \xdd\x92\x77\xb0\x94\x5b\xe2\xaa";
        haraka512::<5>(&mut dst, &src);
        assert_eq!(&dst, expect);
    }

    #[test]
    fn test_haraka512_6round() {
        let mut dst = [0; 32];
        let src = b"\x00\x01\x02\x03\x04\x05\x06\x07\
                    \x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
                    \x10\x11\x12\x13\x14\x15\x16\x17\
                    \x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\
                    \x20\x21\x22\x23\x24\x25\x26\x27\
                    \x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\
                    \x30\x31\x32\x33\x34\x35\x36\x37\
                    \x38\x39\x3a\x3b\x3c\x3d\x3e\x3f";
        let expect = b"\x0e\x27\x51\x4e\x8a\xb7\xb4\xee\
                       \x15\x3c\x9a\x54\x13\xfb\x1e\x98\
                       \x4a\x91\x4f\x5b\x6f\xea\x17\x22\
                       \x85\x41\xce\x17\x07\xfc\x4e\x64";
        haraka512::<6>(&mut dst, &src);
        assert_eq!(&dst, expect);
    }

    use std::hint::black_box;
    use test::Bencher;

    fn haraka512_through<const N_ROUNDS: usize>(src: &[u8; 64]) -> [u8; 32] {
        let mut dst = [0; 32];
        haraka512::<N_ROUNDS>(&mut dst, src);
        dst
    }

    #[bench]
    fn bench_haraka512_5round(b: &mut Bencher) {
        let src = b"\x00\x01\x02\x03\x04\x05\x06\x07\
                    \x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
                    \x10\x11\x12\x13\x14\x15\x16\x17\
                    \x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\
                    \x20\x21\x22\x23\x24\x25\x26\x27\
                    \x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\
                    \x30\x31\x32\x33\x34\x35\x36\x37\
                    \x38\x39\x3a\x3b\x3c\x3d\x3e\x3f";
        b.iter(|| haraka512_through::<5>(black_box(&src)));
    }

    #[bench]
    fn bench_haraka512_6round(b: &mut Bencher) {
        let src = b"\x00\x01\x02\x03\x04\x05\x06\x07\
                    \x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
                    \x10\x11\x12\x13\x14\x15\x16\x17\
                    \x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\
                    \x20\x21\x22\x23\x24\x25\x26\x27\
                    \x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\
                    \x30\x31\x32\x33\x34\x35\x36\x37\
                    \x38\x39\x3a\x3b\x3c\x3d\x3e\x3f";
        b.iter(|| haraka512_through::<6>(black_box(&src)));
    }
}
