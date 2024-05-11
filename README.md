# Drillx

Drillx is a memory hard hash function for Ore cryptocurrency mining.

## Summary

The basic idea is to shift ~99% of the hashing work into simply building the digest and make this a long, non-parallelizable, memory bound process. Drillx borrows a number of ideas from RandomX including the use of a large scratchspace, random instruction execution, non-invertible processes, and unpredictable memory access patterns. However drillx is unique in that it is designed specifically to operate within the constraints of the Solana runtime.

## GPU

To build with the `gpu` feature enabled, you must have CUDA installed on your local machine. You can download the CUDA Toolkit from NVIDIA’s official [CUDA Toolkit website](https://developer.nvidia.com/cuda-downloads). Choose the version compatible with your system and graphics card.


