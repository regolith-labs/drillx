#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "hashx/src/context.h"

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint8_t *out) {
    // Allocate a 40-byte buffer
    uint8_t seed[40];
    memcpy(seed, challenge, 32);
    memcpy(seed + 32, nonce, 8);

    // Create an equix context
    equix_ctx* ctx = equix_alloc(EQUIX_CTX_SOLVE);
    if (ctx == nullptr) {
        printf("Failed to allocate equix context\n");
        return;
    }

    // Make hashx function
	  if (!hashx_make(ctx->hash_func, seed, 40)) {
	      return;
	  }

    // Launch kernel to parallelize hashx operations
    dim3 threadsPerBlock(256); // 256 threads per block
    dim3 blocksPerGrid((65536 + threadsPerBlock.x - 1) / threadsPerBlock.x); // enough blocks to cover 65536 threads
    do_solve_stage0<<<blocksPerGrid, threadsPerBlock>>>(ctx->hash_func, ctx->heap);
    cudaDeviceSynchronize();

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

__global__ void do_solve_stage0(hashx_ctx* hash_func, solver_heap* heap) {
    uint16_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) {
        hash_stage0i(hash_func, heap, i);
    }
}

