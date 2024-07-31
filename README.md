# Drillx

Drillx is a proof-of-work algorithm for smart contract based cryptocurrency mining.

## Summary

Drillx builds upon Equix, the CPU-friendly [client puzzle](https://gitlab.torproject.org/tpo/core/tor/-/blob/main/src/ext/equix/devlog.md) designed to protect Tor from DOS attacks. Equix itself is a variation of [Equihash](https://core.ac.uk/download/pdf/31227294.pdf), an asymmetric proof-of-work function with cheap verifications. Drillx adds a Keccak hashing step on top of Equix to guarantee a difficulty distribution of `p(Z) = 2^-Z` where `Z` is the number of leading zeros on the hash. A challenge `C` is assumed to be a securely generated 256-bit hash, seeded by a recent Solana blockhash. Miners compete to find a 64-bit nonce `N` that produces a hash of their target difficulty. Solutions must be presented in the form `(E, N)` where `E = Equix(C, N)`. The difficulty is calculated from `Keccak(E', N)` where `E'` is the Equix proof, sorted lexographically to prevent malleability. Since `E` can be efficiently verified on-chain and Keccak is available as a Solana syscall, Drillx solutions can easily fit into a single Solana transaction.

## Usage
Miners can iterate through nonces to find a hash that satisfies their target difficulty.
```rs
use drillx::{equix::SolverMemory, Solution};

fn main() {
    let challenge = [255; 32]; // Should be provided by a program
    let target = 8;
    let mut memory = SolverMemory::new();
    for nonce in 0..u64::MAX {
        let hx = drillx::hash_with_memory(&mut memory, &challenge, &nonce.to_le_bytes());
        if hx.difficuty() >= target {
            println!("Solution: {:?}", Solution::new(hx.d, nonce));
            return
        }
    }
}
```

Smart contracts can verify the solution and use the difficulty to issue token rewards.
```rs
use drillx::Solution;

fn verify(solution: Solution) {
    let challenge = [255; 32]; // Fetch from state
    let target = 8;
    assert!(solution.is_valid(&challenge));
    assert!(solution.to_hash().difficulty() >= target);
}
```
