# Drillx

Drillx is an asic-resistant hashing algorithm for Ore cryptocurrency mining.

## Basic idea
There are two steps to creating any hash:
1. Build the digest
2. Do the hash


In Ore v1, building the digest was trivially easy. 100% of the work was the keccak hash. The basic idea of drillx is to shift ~90-99% of the work to building the digest and make this process a long a non-parallelizable, memory bound task.

Drillx borrows many ideas from RandomX including the use of a large scratchspace, random instruction execution, non-invertible processes, and unpredictable memory access patterns. However drillx is fine tuned to operate within the limits of the Solana runtime and execute a hash within a single Solana transaction.
