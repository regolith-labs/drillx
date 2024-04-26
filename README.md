# Drillx

Drillx is an asic-resistant hashing algorithm for Ore cryptocurrency mining.

## Basic idea
The basic idea of drillx is to shift ~99% of the hashing work to building the digest and make this a long a non-parallelizable, memory bound process. Drillx borrows many ideas from RandomX including the use of a large scratchspace, random instruction execution, non-invertible processes, and unpredictable memory access patterns. However drillx is specifically designed to operate within the constraints of the Solana runtime.
