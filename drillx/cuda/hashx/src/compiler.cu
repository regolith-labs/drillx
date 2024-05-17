/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#include <stdlib.h>
#include <stdbool.h>

#include "compiler.h"
#include "program.h"
#include "context.h"

__device__ bool hashx_compiler_init(hashx_ctx* ctx) {
	ctx->code = (uint8_t*)malloc(COMP_CODE_SIZE);
    if (ctx->code == NULL) {
    	return false;
    }
    return true;
}

__device__ void hashx_compiler_destroy(hashx_ctx* ctx) {
	if (ctx->code != NULL) {
        free(ctx->code);
        ctx->code = NULL;
    }
}
