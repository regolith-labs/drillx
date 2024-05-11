# Drillx

Drillx is a memory hard hash function for Ore cryptocurrency mining.

## Summary

The basic idea of Drillx is to make building the correct Keccak digest harder than the hash itself. This process should be long, non-parallelizable, and I/O bound. Drillx additionally borrows a number of ideas from RandomX including the use of a large scratchspace, random instruction execution, and unpredictable memory access patterns. It adapts these mechanics to make them fit within the limited constraints of the Solana runtime.

## GPU

To run drillx on a CUDA compatible gpu, compile the crate with the `gpu` feature flag enabled. You must have CUDA installed on your local machine. You can download the CUDA toolkit from [NVIDIA’s official website](https://developer.nvidia.com/cuda-downloads). Choose the version compatible with your system and graphics card.


