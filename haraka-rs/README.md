# haraka-rs

![Build Status](https://github.com/gendx/haraka-rs/workflows/Build/badge.svg)
![Test Status](https://github.com/gendx/haraka-rs/workflows/Tests/badge.svg)

A Rust implementation of the [Haraka](https://github.com/kste/haraka) (v2) short-input hash function.

## Implementation

As for the original Haraka implementation in C, this project relies on AES-NI instructions, which are available on the stable Rust compiler via [intrinsics](https://doc.rust-lang.org/core/arch/x86_64/fn._mm_aesenc_si128.html).

Besides the original 5-round Haraka functions (with 256 and 512 bits of input), extensions to 6 rounds are provided.
This is to target collision resistance, contrary to the 5-round versions that only provide preimage resistance.

## Testing

Unit tests are implemented to check the logic of Haraka's building blocks.
High-level test vectors were generated from the [Python implementation](https://github.com/kste/haraka/blob/master/code/python/ref.py) of Haraka (for the 5-round versions).

## License

MIT

