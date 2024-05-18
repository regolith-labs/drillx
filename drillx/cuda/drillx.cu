#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "hashx/src/context.h"

// TODO Add batch size parameter. We will do hashes for (nonce..nonce+batch_size)
extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint8_t *out) {
    // TODO Generate seeds based on batch size
    // Allocate a 40-byte buffer
    uint8_t seed[40];
    memcpy(seed, challenge, 32);
    memcpy(seed + 32, nonce, 8);

    // TODO Allocate array INDEX_SPACE large for hashx values
    // TODO One array for each hash we are going to offload to gpu
    // Create an equix context
    equix_ctx* ctx = equix_alloc(EQUIX_CTX_SOLVE);
    if (ctx == nullptr) {
        printf("Failed to allocate equix context\n");
        return;
    }

    // TODO Generate hash function for each seed
    // Make hashx function
	  if (!hashx_make(ctx->hash_func, seed, 40)) {
	      return;
	  }

    // TODO Figure out how many threads and blocks we need based on the batch size
    // TODO Pass all addresses to the kernel
    // Launch kernel to parallelize hashx operations
    dim3 threadsPerBlock(256); // 256 threads per block
    dim3 blocksPerGrid((65536 + threadsPerBlock.x - 1) / threadsPerBlock.x); // enough blocks to cover 65536 threads
    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock>>>(ctx->hash_func, ctx->heap);
    cudaDeviceSynchronize();

    // TODO Return data and spawn thread to do the remaining stages from rust (cpu)
    // Do the remaining stages
    equix_solution solutions[EQUIX_MAX_SOLS];
    int num_sols = equix_solver_solve(ctx->heap, solutions);

    // Copy results back to host
    if (num_sols > 0) {
        memcpy(out, solutions[0].idx, sizeof(solutions[0].idx));
    }

    // Free memory
    equix_free(ctx);

    // Print errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// TODO Pick correct heap, hash_func, and i 
// TODO Do many i per thread
__global__ void do_hash_stage0i(hashx_ctx* hash_func, solver_heap* heap) {
    uint16_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) {
        hash_stage0i(hash_func, heap, i);
    }
}

