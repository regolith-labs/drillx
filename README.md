# Drillx

Drillx is an asic-resistant hashing algorithm for Ore cryptocurrency mining.

## Summary
The basic idea of drillx is to shift ~99% of the hashing work into simply building the digest and make this a long, non-parallelizable, memory bound process. Drillx borrows many ideas from RandomX including the use of a large scratchspace, random instruction execution, non-invertible processes, and unpredictable memory access patterns. However drillx is unique in that it is specifically designed to operate within the constraints of the Solana runtime.
