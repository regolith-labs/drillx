#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "hashx/src/context.h"

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint8_t *out) {
    // Allocate device memory for input and output data
    uint8_t *d_challenge, *d_nonce, *d_out;
    cudaMalloc((void **)&d_challenge, 32);
    cudaMalloc((void **)&d_nonce, 8);
    cudaMalloc((void **)&d_out, 16);
	  cudaMemcpy(d_challenge, challenge, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce, nonce, 8, cudaMemcpyHostToDevice);

    // Create an equix context
    equix_ctx* ctx = equix_alloc(EQUIX_CTX_SOLVE);
    if (ctx == nullptr) {
        printf("Failed to allocate equix context\n");
        return;
    }

    // Make hashx function
	  if (!hashx_make(ctx->hash_func, challenge, 32)) {
	  	return;
	  }

    // Launch kernel to parallelize hashx operations
    dim3 threadsPerBlock(256); // 256 threads per block
    dim3 blocksPerGrid((65536 + threadsPerBlock.x - 1) / threadsPerBlock.x); // enough blocks to cover 65536 threads
    do_solve_stage0<<<blocksPerGrid, threadsPerBlock>>>(ctx->hash_func, ctx->heap);
    cudaDeviceSynchronize();

    // TODO Do the remaining stages
    equix_solution* output = (equix_solution*)malloc(EQUIX_MAX_SOLS * sizeof(equix_solution));
    if (output == NULL) {
        return;
    }
    int sols = solve_stage123(ctx->heap, output);
    printf("sols %d", sols);

    // Free equix context
    equix_free(ctx);

    // Copy results back to host
    cudaMemcpy(out, d_out, 16, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_challenge);
    cudaFree(d_nonce);
    cudaFree(d_out);

    // Print errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void do_solve_stage0(hashx_ctx* hash_func, solver_heap* heap) {
    uint16_t i = blockIdx.x * blockDim.x + threadIdx.x;
    prep_stage0(heap);
    if (i < 65536) {
        solve_stage0i(hash_func, heap, i);
    }
}

