# Original VerusCoin C Implementation Files

This directory (`origin-impl`) contains the original C/C++ source code files related to VerusHash, directly copied from the VerusCoin repository:

[https://github.com/VerusCoin/VerusCoin/tree/master/src/crypto](https://github.com/VerusCoin/VerusCoin/tree/master/src/crypto)

These files serve as the reference implementation.

## Relationship to `verus/c` Directory

The C/C++ files located in the `../verus/c` directory are **derived** from the files in this `origin-impl` directory. The primary difference is that the files in `verus/c` have been **modified and adapted** to be compatible with the Solana **BPF (Berkeley Packet Filter)** environment.

The goal was to create a version of the VerusHash algorithm that could be executed efficiently and correctly within a Solana smart contract (program) on-chain. This required several changes, including:

1.  **Removing Non-BPF Compatible Code:** Eliminating dependencies on standard libraries (libc, libstdc++), system calls, dynamic memory allocation, and certain CPU intrinsics (like AES-NI, although a portable software AES implementation is used).
2.  **Providing Stubs/Replacements:** Creating minimal implementations or stubs for functions or data structures that were needed but not available in the BPF environment (e.g., `memcpy`, `memset`, simplified `uint256`).
3.  **Build System Integration:** Adapting the build process (`build.rs`, `build.sh`) to compile these C/C++ files into a static library (`libverushash.a`) suitable for linking into the Rust-based Solana program.
4.  **Constant Generation:** Implementing a build-time mechanism (`generate_constants.c`) to pre-compute Haraka round constants, avoiding runtime initialization issues within BPF.

## File Differences

You may notice that some files exist in `verus/c` that are not present here (e.g., `generate_constants.c`, `common.h`, `verus_clhash.h`). These files were specifically created during the adaptation process to meet the requirements of the BPF environment and the build system. Conversely, some files from the original implementation might be omitted from `verus/c` if they were not needed for the core hashing logic required by the Solana program.

