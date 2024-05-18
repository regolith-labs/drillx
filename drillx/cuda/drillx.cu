#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

const int BATCH_SIZE = 512;

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    // Generate a hash function for each (challenge, nonce)
    hashx_ctx** ctxs;
    if (cudaMallocManaged(&ctxs, BATCH_SIZE * sizeof(hashx_ctx*)) != cudaSuccess) {
        printf("Failed to allocate managed memory for ctxs\n");
        return;
    }
    uint8_t seed[40];
    memcpy(seed, challenge, 32);
    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed + 32, &nonce_offset, 8);
        ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!ctxs[i] || !hashx_make(ctxs[i], seed, 40)) {
            printf("Failed to make hash\n");
            return;
        }
    }

    // Allocate space to hold on to hash values (~500KB per seed)
    uint64_t** hash_space;
    if (cudaMallocManaged(&hash_space, BATCH_SIZE * sizeof(uint64_t*)) != cudaSuccess) {
        printf("Failed to allocate managed memory for hash_space\n");
        return;
    }
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (cudaMallocManaged(&hash_space[i], INDEX_SPACE * sizeof(uint64_t)) != cudaSuccess) {
            printf("Failed to allocate managed memory for hash_space[%d]\n", i);
            return;
        }
    }

    // Launch kernel to parallelize hashx operations
    dim3 threadsPerBlock(256); // 256 threads per block
    dim3 blocksPerGrid((65536 * BATCH_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x); // enough blocks to cover batch
    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock>>>(ctxs, hash_space);
    cudaDeviceSynchronize();

    // Copy hashes back to cpu
    for (int i = 0; i < BATCH_SIZE; i++) {
        cudaMemcpy(out + i * INDEX_SPACE, hash_space[i], INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    // Free memory
    for (int i = 0; i < BATCH_SIZE; i++) {
        hashx_free(ctxs[i]);
        cudaFree(hash_space[i]);
    }
    cudaFree(hash_space);
    cudaFree(ctxs);

    // Print errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space) {
    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = item / INDEX_SPACE;
    uint32_t i = item % INDEX_SPACE;
    if (batch_idx < BATCH_SIZE) {
        hash_stage0i(ctxs[batch_idx], hash_space[batch_idx], i);
    }
}

extern "C" void solve_all_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols) {
    // Create an equix context
    equix_ctx* ctx = equix_alloc(EQUIX_CTX_SOLVE);
    if (ctx == nullptr) {
        printf("Failed to allocate equix context\n");
        return;
    }

    // Do the remaining stages
    equix_solution solutions[EQUIX_MAX_SOLS];
    uint32_t num_sols = equix_solver_solve(hashes, ctx->heap, solutions);

    // Copy results back to host
    memcpy(sols, &num_sols, sizeof(num_sols));
    if (num_sols > 0) {
        memcpy(out, solutions[0].idx, sizeof(solutions[0].idx));
    }

    // Free memory
    equix_free(ctx);
}
