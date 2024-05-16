# Drillx

Drillx is a proof-of-work algorithm for smart-contract based cryptocurrency mining.

## Summary

Drillx builds upon Equix, the CPU-friendly [client puzzle](https://gitlab.torproject.org/tpo/core/tor/-/blob/main/src/ext/equix/devlog.md) designed to protect Tor from DOS attacks. Equix itself is a variation of [Equihash](https://core.ac.uk/download/pdf/31227294.pdf), an asymmetric proof-of-work function with cheap verifications. Drillx adds a Blake3 hashing step on top of Equix to guarantee a difficulty distribution of `p(Z) = 2^-Z` where `Z` is the number of leading zeros on the hash. A challenge `C` is assumed to be a securely generated 256-bit hash, seeded by a recent Solana blockhash. Miners compete to find a 64-bit nonce `N` that produces a hash of their target difficulty. Solutions must be presented in the form `(E, N)` where `E = Equix(C, N)`. The difficulty is calculated from `Blake3(E', N)` where `E'` is the Equix proof, sorted lexographically to prevent malleability. Since `E` can be efficiently verified on-chain and Blake3 is available as a Solana syscall, Drillx solutions can easily fit into a single Solana transaction.

## Usage
```rs
use drillx::equix::SolverMemory;

fn main() {
    let challenge = [255; 32];
    for nonce in 0..u64::MAX {
        let _hx = drillx::hash_with_memory(&mut memory, &challenge, &nonce.to_le_bytes())
    }
}
```
