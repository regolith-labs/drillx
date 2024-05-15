# Drillx

Drillx is a CPU-friendly proof-of-work function for Ore cryptocurrency mining.

## Summary

Drillx builds upon Equix, the [client puzzle](https://gitlab.torproject.org/tpo/core/tor/-/blob/main/src/ext/equix/devlog.md) created to protect the Tor network from DOS attacks. Equix itself is a variation of [Equihash](https://core.ac.uk/download/pdf/31227294.pdf), an asymmetric proof-of-work function with cheap verification. Drillx adds a Blake3 hashing step on top of Equix to provide a guaranteed difficulty distribution of `p(Z) = 2^-Z` where `Z` is the number of leading zeros on the hash. A challenge `C` is assumed to be securely generated 256-bit hash, seeded by a recent Solana blockhash. Miners aim to find a 64-bit nonce `N` that produces a hash of their target difficulty. Solutions must be presented in the form `(E, N)` where `E = Equix(C, N)`. The difficulty is calculated from `Blake3(E', N)` where `E'` is the Equix hash, lexographically sorted to prevent malleability. Since `E` can be efficiently verified on chain and Blake3 is available as a Solana syscall, Drillx is ideally suited to serve as a proof-of- work function for Ore cryptocurrency mining. 
