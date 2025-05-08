Portable no-std Haraka fork for eBPF & Solana SBF

# haraka-bpf

A fork of [haraka-rs](https://github.com/gendx/haraka-rs), providing a portable, software-only Rust implementation of the [Haraka](https://github.com/kste/haraka) (v2) short-input hash function, suitable for BPF/SBF environments like the Solana runtime.

## Implementation

This fork diverges from the original `haraka-rs` by removing the dependency on hardware AES-NI instructions. Instead, it utilizes the [aes](https://crates.io/crates/aes) crate for a pure software implementation of AES rounds.

This makes the crate portable and allows it to be compiled to targets like BPF (specifically SBF for the Solana runtime) where hardware intrinsics are not available.

The implementation provides the original 5-round Haraka functions (for 256 and 512 bits of input) which offer preimage resistance, as well as extensions to 6 rounds targeting collision resistance.

## Building

You can compile this crate for both Solana SBF and generic Linux/eBPF targets using the same source:

### Solana SBF

```bash
# Build using the Solana toolchain wrapper
cargo build-sbf
```

### Linux eBPF (generic BPF)

Make sure you have the Rust source component for nightly:

```bash
rustup +nightly component add rust-src
```

Then compile core and alloc for the BPF target:

```bash
cargo +nightly build \
  --target bpfel-unknown-none \
  --release \
  --no-default-features \
  -Zbuild-std=core,alloc
```

## Testing

Unit tests are implemented to check the logic of Haraka's building blocks.
High-level test vectors were generated from the [Python implementation](https://github.com/kste/haraka/blob/master/code/python/ref.py) of Haraka (for the 5-round versions).

## License

MIT
