#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    // Generate a hash function for each (challenge, nonce)
    hashx_ctx* ctxs[BATCH_SIZE];
    uint8_t seed[40];
    memcpy(seed, challenge, 32);
    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed + 32, &nonce_offset, 8);
        ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!hashx_make(ctxs[i], seed, 40)) {
            // TODO Handle error
            printf("Failed to make hash\n");
            return;
        }
    }

    // Allocate space to hold on to hash values (~500KB per seed)
    // printf("C");
    // uint64_t* hash_space;
    // size_t total_size = BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t);
    // cudaMalloc((void**)&hash_space, total_size);
    uint64_t* hash_space[INDEX_SPACE];
    size_t total_size = BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t);
    cudaMalloc((void**)&hash_space, total_size);

    // Launch kernel to parallelize hashx operations
    // printf("D");
    dim3 threadsPerBlock(256); // 256 threads per block
    dim3 blocksPerGrid((65536 * BATCH_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x); // enough blocks to cover batch
    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock>>>(ctxs, hash_space);
    cudaDeviceSynchronize();

    // Copy hashes back to cpu
    // printf("E");
    cudaMemcpy(out, hash_space, total_size, cudaMemcpyDeviceToHost);

    // Free memory
    for (int i = 0; i < BATCH_SIZE; i++) {
        hashx_free(ctxs[i]);
    }
    cudaFree(hash_space);

    // Print errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void do_hash_stage0i(hashx_ctx* ctxs[BATCH_SIZE], uint64_t* hash_space[INDEX_SPACE]) {
    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = item / INDEX_SPACE;
    uint32_t i = item % INDEX_SPACE;
    if (batch_idx < BATCH_SIZE) {
        hash_stage0i(ctxs[batch_idx], hash_space[batch_idx * INDEX_SPACE], i);
    }
}

// TODO
// extern "C" void solve(uint8_t *hashes, uint8_t *out) {
    // Create an equix context
    // equix_ctx* ctx = equix_alloc(EQUIX_CTX_SOLVE);
    // if (ctx == nullptr) {
    //     printf("Failed to allocate equix context\n");
    //     return;
    // }

    // Do the remaining stages
    // equix_solution solutions[EQUIX_MAX_SOLS];
    // int num_sols = equix_solver_solve(ctx->heap, solutions);

    // Copy results back to host
    // if (num_sols > 0) {
    //     memcpy(out, solutions[0].idx, sizeof(solutions[0].idx));
    // }

    // Free memory
    // equix_free(ctx);
// }
