#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix.h"

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint8_t *out) {
    // Allocate device memory for input and output data
    uint8_t *d_challenge, *d_nonce, *d_out;
    cudaMalloc((void **)&d_challenge, 32);
    cudaMalloc((void **)&d_nonce, 8);
    cudaMalloc((void **)&d_out, 16);
	cudaMemcpy(d_challenge, challenge, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce, nonce, 8, cudaMemcpyHostToDevice);

    // Launch kernel
    do_hash<<<1, 1>>>(d_challenge, d_nonce, d_out);

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

__global__ void do_hash(uint8_t *d_challenge, uint8_t *d_nonce, uint8_t *d_out) {
    // Create an equix context
    equix_ctx* ctx = equix_alloc(EQUIX_CTX_SOLVE);
    if (ctx == nullptr) {
        printf("Failed to allocate equix context\n");
        return;
    }

    // Create a solutions buffer
    equix_solution solutions[EQUIX_MAX_SOLS];
    int num_solutions = equix_solve(ctx, d_challenge, 32, solutions);

    // Check the result
    // TODO Output full solution
    if (num_solutions > 0) {
        *d_out = solutions[0].idx[0];
    } else {
        *d_out = 42;
    }

    // Free the equix context
    equix_free(ctx);
}

