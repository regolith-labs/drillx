/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#include <stdlib.h>
#include <stdbool.h>

#include "compiler.h"
#include "program.h"
#include "context.h"

__device__ bool hashx_compiler_init(hashx_ctx* ctx) {
	if (cudaMalloc(&ctx->code, COMP_CODE_SIZE) != cudaSuccess) {
        return false; // Allocation failed
    }
    return true;
}

__device__ void hashx_compiler_destroy(hashx_ctx* ctx) {
	if (ctx->code != NULL) {
        cudaFree(ctx->code);
        ctx->code = NULL;
    }
}
